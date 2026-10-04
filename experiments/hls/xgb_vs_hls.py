from __future__ import annotations

from puppibuff import from_zip
from puppibuff.analyses import plot_histograms
from puppibuff.cli.common import timed
from puppibuff.cli.hls import build_hls
from puppibuff.hls import constants
from puppibuff.utils import initial_noise, output_dir

import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

N_SAMPLES  = 1_000_000
N_HLS      =    50_000


def main():                             # HLS project directory and the trained
    if len(sys.argv) < 3:               # archive both paths sample from
        sys.exit(f"Usage: {sys.argv[0]} <workdir> <model>")

    workdir = sys.argv[1]

    config, codec, model = from_zip(sys.argv[2])
    data = config.dataset()

    hls = build_hls(model, codec, workdir)

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

    path = output_dir(__file__) / f"{Path(workdir).name}_xgb_vs_hls.pdf"
    figure.savefig(path, format = "pdf")
    print(f"Wrote {path}", flush = True)


if __name__ == "__main__":
    main()
