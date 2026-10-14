from puppibuff.analyses import plot_histograms
from puppibuff.configs import MultiplicityL1PuppiConfig

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

def main():
    config = MultiplicityL1PuppiConfig(n_steps = 4)
    config.n_events = 100_000
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2


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
