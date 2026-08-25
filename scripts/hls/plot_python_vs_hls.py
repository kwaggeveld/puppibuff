from __future__ import annotations

from puppibuff.analyses import plot_histograms
from puppibuff.utils import from_zip

import sys
from pathlib import Path

import numpy as np

#-----------------------------------------------------------------------------

# Plotting of output of `scripts/hls/sample.py`

def main():                             # The `.npz` `sample.py` wrote, and the
    if len(sys.argv) < 3:               # model it sampled from
        sys.exit(f"Usage: { sys.argv[0] } <samples> <model>")

    samples = np.load(sys.argv[1])

    config, codec, _ = from_zip(sys.argv[2])

                                        # HLS takes primary slot, ratios read
                                        # HLS/target and HLS/xgboost
    figure = plot_histograms(
        config.dataset(), 
        sample  = codec.decode(samples["hls"]),
        overlay = codec.decode(samples["xgb"]),
        labels  = { "Output": "HLS", "Training": "Python" },
    )

    outdir = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    outdir.mkdir(parents = True, exist_ok = True)

    path = outdir / f"{ Path(sys.argv[1]).stem }.pdf"
    figure.savefig(path, format = "pdf")

    print(f"Wrote { path }")


if __name__ == "__main__":
    main()
