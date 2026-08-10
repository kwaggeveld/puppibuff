from puppibuff.analyses.plotting import plot_histograms
from puppibuff.configs import ClusteredL1PuppiConfig


def main():
                                        # One BDT per (step, channel), each
                                        # predicting that channel for all slots
    config = ClusteredL1PuppiConfig(multi_output = True)
    config.n_events = 500_000
    config.tree_config["n_estimators"] = 100
    config.tree_config["max_depth"] = 4


    data, codec, model, x, y = config.setup()

    model.fit(x, y)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_histograms(
        data, samples, n_events = config.n_events,
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
