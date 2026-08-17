from puppibuff.analyses import plot_contours
from puppibuff.utils import from_zip

import sys
from pathlib import Path

#-----------------------------------------------------------------------------

# Pairwise KDE contour plots, like BUFF's Fig. 2 (arXiv:2404.18219).

N_SAMPLES = 500_000

def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <model>")

    config, codec, model = from_zip(sys.argv[1])

    data    = config.dataset()
    samples = codec.decode(model.sample(N_SAMPLES))

    figure = plot_contours(data, samples)

    outdir = Path(__file__).resolve().parent / "output" / Path(__file__).stem
    outdir.mkdir(parents = True, exist_ok = True)

    file = outdir / f"contours_{ Path(sys.argv[1]).stem }.pdf"
    figure.savefig(file, format = "pdf")

    print(f"Wrote { file }")


if __name__ == "__main__":
    main()
