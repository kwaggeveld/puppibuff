from puppibuff.analyses.plotting import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.hls import FlowHLS

import sys
from pathlib import Path

N_SAMPLES  = 1_000_000
N_HLS      =    50_000

MERGED     = True

def main():                             # Pass directory for the HLS project(s)
    if len(sys.argv) < 2:               # as argument 1
        sys.exit(f"Usage: {sys.argv[0]} [workdir]")

    workdir = sys.argv[1]

    config = FlatPuppiJetConfig(n_steps = 15,
                                n_events = 2_000_000)
    config.tree_config["n_estimators"] = 50
    config.tree_config["max_depth"] = 4

    data, codec, model, x, y = config.setup()

    model.fit(x, y)

    hls = FlowHLS.convert(model, output_dir = workdir, merged = MERGED)
    hls.write(codec)
    hls.compile(n_threads = 7)

    sample = hls.sample(N_HLS)

    figure = plot_histograms(
        data, codec.decode(sample), n_events = config.n_events,
    )

    path = Path(workdir) / f"{Path(workdir).name}_hls.pdf"
    figure.savefig(path, format = "pdf")


if __name__ == "__main__":
    main()
