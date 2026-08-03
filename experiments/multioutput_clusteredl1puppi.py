from puppibuff.analyses.plotting import plot_distributions_jet
from puppibuff.configs import ClusteredL1PuppiConfig

from puppibuff.utils import setup_from_config




def main():
                                        # One BDT per (step, channel), each
                                        # predicting that channel for all slots
    config = ClusteredL1PuppiConfig(multi_output = True,
                                    s1phi = False)
    config.n_events = 500_000
    config.tree_config["n_estimators"] = 100
    config.tree_config["max_depth"] = 4


    data, codec, model, x, y = setup_from_config(config)

    model.fit(x, y)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_distributions_jet(
        data, samples, channels = ["pt", "eta", "phi"], n_events = config.n_events,
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
