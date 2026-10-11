from puppibuff.analyses import plot_histograms
from puppibuff.configs import ClusteredL1PuppiConfig

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

def main():
    config = ClusteredL1PuppiConfig()
    config.n_events = 500_000

    data, codec, flowbdt, x, y = config.setup()

    flowbdt.fit(x, y)

    raw_samples = flowbdt.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_histograms(
        data, { "Output": samples, "Training": data[:config.n_events] },
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
