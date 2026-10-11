from puppibuff.analyses import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.utils import output_dir

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

def main():
    config = FlatPuppiJetConfig()

    data, codec, flowbdt, x, y = config.setup()

    flowbdt.fit(x, y)

    raw_samples = flowbdt.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_histograms(
        data, { "Output": samples, "Training": data[:config.n_events] },
    )

    figure.savefig(output_dir(__file__) / "puppijet_full_d6_n50.pdf")


if __name__ == "__main__":
    main()
