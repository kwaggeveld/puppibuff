from puppibuff import from_zip
from puppibuff.analyses import plot_histograms
from puppibuff.utils import output_dir

import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

# Histogram plots from pretrained model

N_SAMPLES = 1_000_000

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <model>")

    config, codec, model = from_zip(sys.argv[1])

    data    = config.dataset()
    samples = codec.decode(model.sample(N_SAMPLES))

    figure = plot_histograms(data, samples)

    file = output_dir(__file__) / f"{ Path(sys.argv[1]).stem }_{ N_SAMPLES :,}.pdf"
    figure.savefig(file, format = "pdf")

    print(f"Wrote { file }")


if __name__ == "__main__":
    main()
