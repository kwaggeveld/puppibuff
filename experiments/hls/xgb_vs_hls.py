from __future__ import annotations

from puppibuff.analyses import plot_histograms
from puppibuff.hls import constants, FlowHLS
from puppibuff.utils import from_zip, initial_noise

import sys
import time
from pathlib import Path

N_SAMPLES  = 1_000_000
N_HLS      =    50_000

MERGED     = True


def timed(label: str, call, *args, **kwargs):
    """Announce a step before it blocks, and report what it cost."""
    print(f"{label}...", flush = True)
    start  = time.time()
    result = call(*args, **kwargs)
    print(f"{label}: {time.time() - start:.1f} s", flush = True)

    return result


def build_hls(model, codec, workdir: str, reuse: bool) -> FlowHLS:
    """Convert and compile the grid, or bind an existing build in `workdir`."""

    if reuse:                           # `load` reads the layout off `workdir`
        return timed("Loading compiled grid", FlowHLS.load, workdir)

    hls = timed("Converting grid", FlowHLS.convert, model,
                output_dir = workdir, merged = MERGED)
    timed("Writing", hls.write, codec)
    timed("Compiling", hls.compile, n_threads = 7)

    return hls


def main():                             # HLS project directory and the trained
    if len(sys.argv) < 3:               # archive both paths sample from
        sys.exit(f"Usage: {sys.argv[0]} <workdir> <model>")

    workdir = sys.argv[1]

    reuse = Path(workdir).exists() and any(Path(workdir).rglob("bdt_s*_g*.json"))

    config, codec, model = from_zip(sys.argv[2])
    data = config.dataset()

    hls = build_hls(model, codec, workdir, reuse)

    outdir = Path(__file__).resolve().parent / Path(__file__).stem
    outdir.mkdir(exist_ok = True)

    x0 = initial_noise((N_SAMPLES, hls.n_channels))

    hls_sample = timed(f"Sampling {N_HLS} with hls", hls.sample,
                       x0 = x0[:N_HLS], solver = constants.SAMPLE_SOLVER)

    xgb_sample = timed(f"Sampling {N_SAMPLES} with xgboost", model.sample,
                       x0 = x0, solver = constants.SAMPLE_SOLVER)

                                        # HLS takes primary slot, ratios read
                                        # HLS/target and HLS/xgboost
    figure = plot_histograms(
        data, codec.decode(hls_sample),
        overlay = codec.decode(xgb_sample),
        labels  = { "Output": "HLS", "Training": "XGBoost" },
    )

    path = outdir / f"{Path(workdir).name}_xgb_vs_hls.pdf"
    figure.savefig(path, format = "pdf")
    print(f"Wrote {path}", flush = True)


if __name__ == "__main__":
    main()
