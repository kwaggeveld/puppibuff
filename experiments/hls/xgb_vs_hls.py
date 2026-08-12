from __future__ import annotations

from puppibuff.analyses.plotting import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import constants, FlowHLS
from puppibuff.utils import initial_noise

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


def main():                             # Pass directory for the HLS project(s)
    if len(sys.argv) < 2:               # as argument 1
        sys.exit(f"Usage: {sys.argv[0]} [workdir]")

    workdir = sys.argv[1]

    reuse = Path(workdir).exists() and any(Path(workdir).rglob("bdt_s*_g*.json"))

    config = FlatPuppiJetConfig(n_steps = 15,
                                n_events = 2_000_000)
    config.tree_config["n_estimators"] = 50
    config.tree_config["max_depth"] = 4

    data, codec, model, x, y = config.setup()

    if not reuse:
        model.fit(x, y)

    hls = build_hls(model, codec, workdir, reuse)

    outdir = Path(__file__).resolve().parent / Path(__file__).stem
    outdir.mkdir(exist_ok = True)

    x0 = initial_noise(n_samples = N_SAMPLES, n_channels = hls.n_channels)

    samplers = [("hls", hls, N_HLS)]
    if not reuse:
        samplers.insert(0, ("xgboost", model, N_SAMPLES))                     # type: ignore

    for label, sampler, n_samples in samplers:
                                        # Pinned, not defaulted: the merged design
                                        # has `constants.SAMPLE_SOLVER` compiled into
                                        # it, so an A/B against XGBoost is only
                                        # apples-to-apples on that same scheme
        sample = timed(f"Sampling {n_samples} with {label}",
                       sampler.sample, x0 = x0[:n_samples],
                       solver = constants.SAMPLE_SOLVER)

        figure = plot_histograms(
            data, codec.decode(sample), n_events = config.n_events,
        )
        figure.suptitle(label)

        path = outdir / f"{Path(workdir).name}_{label}.pdf"
        figure.savefig(path, format = "pdf")
        print(f"Wrote {path}", flush = True)


if __name__ == "__main__":
    main()
