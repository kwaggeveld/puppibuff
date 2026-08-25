from puppibuff.analyses import plot_histograms
from puppibuff.utils import from_zip

import sys
from pathlib import Path

#-----------------------------------------------------------------------------

# Histogram plots from pretrained model

N_SAMPLES = 500_000

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <model>")

    config, codec, model = from_zip(sys.argv[1])

    data    = config.dataset()
    samples = codec.decode(model.sample(N_SAMPLES))

    figure = plot_histograms(data, samples, n_events = config.n_events, bins = 50)

    outdir = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    outdir.mkdir(parents = True, exist_ok = True)

    file = outdir / f"{ Path(sys.argv[1]).stem }.pdf"
    figure.savefig(file, format = "pdf")

    print(f"Wrote { file }")


if __name__ == "__main__":
    main()
