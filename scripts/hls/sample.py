from puppibuff.hls import constants, FlowHLS
from puppibuff.utils import from_zip, initial_noise

import sys
import time
from pathlib import Path

import numpy as np

#-----------------------------------------------------------------------------

# Sample a trained grid through both HLS and XGBoost, and save the two *encoded* 
# samples for `scripts/hls/plot_xgb_vs_hls.py` to draw.

N_SAMPLES = 1_000_000                   # XGBoost
N_HLS     = 1_000_000                   # HLS

MERGED    = True


def timed(label: str, call, *args, **kwargs):
    """Announce a step before it blocks, and report what it cost."""
    print(f"{ label }...", flush = True)
    start  = time.time()
    result = call(*args, **kwargs)
    print(f"{ label }: { time.time() - start :.1f} s", flush = True)

    return result


def build_hls(model, codec, workdir: str) -> FlowHLS:
    """Convert and compile the grid, or bind an existing build in `workdir`."""
                                        # `load` reads the layout off `workdir`
    if Path(workdir).exists() and any(Path(workdir).rglob("bdt_s*_g*.json")):
        return timed("Loading compiled grid", FlowHLS.load, workdir)

    hls = timed("Converting grid", FlowHLS.convert, model,
                output_dir = workdir, merged = MERGED)
    timed("Writing", hls.write, codec)
    timed("Compiling", hls.compile, n_threads = 7)

    return hls


def main():                             # HLS project directory and the trained
    if len(sys.argv) < 3:               # archive both paths sample from
        sys.exit(f"Usage: { sys.argv[0] } <workdir> <model>")

    workdir = sys.argv[1]

    _, codec, model = from_zip(sys.argv[2])

    hls = build_hls(model, codec, workdir)

    x0 = initial_noise((N_SAMPLES, hls.n_channels))

    hls_sample = timed(f"Sampling { N_HLS } with hls", hls.sample,
                       x0 = x0[:N_HLS], solver = constants.SAMPLE_SOLVER)

    xgb_sample = timed(f"Sampling { N_SAMPLES } with xgboost", model.sample,
                       x0 = x0, solver = constants.SAMPLE_SOLVER)

    path = Path(workdir) / f"{ Path(workdir).name }_samples.npz"

                                        # Saved encoded
    np.savez(path, hls = hls_sample.astype(np.float32),
                   xgb = xgb_sample.astype(np.float32))

    print(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB)", flush = True)


if __name__ == "__main__":
    main()
