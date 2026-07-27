from puppibuff.analyses.plotting import plot_distributions_jet
from puppibuff.configs import MultiplicityL1PuppiConfig

from puppibuff import setup_from_config



def main():
    config = MultiplicityL1PuppiConfig(multi_output = False,
                                       s1phi = False,
                                       n_steps = 4)
    config.n_events = 100_000
    config.tree_config["n_estimators"] = 20
    config.tree_config["max_depth"] = 2


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
