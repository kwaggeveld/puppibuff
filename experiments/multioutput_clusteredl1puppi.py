from puppibuff.analyses import plot_histograms
from puppibuff.configs import ClusteredL1PuppiConfig

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

def main():
                                        # One BDT per (step, channel), each
                                        # predicting that channel for all slots
    config = ClusteredL1PuppiConfig(multi_output = True)
    config.n_events = 500_000
    config.tree_config["n_estimators"] = 100
    config.tree_config["max_depth"] = 4


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
