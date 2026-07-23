from puppibuff.analyses.plotting import plot_distributions_jet
from puppibuff.configs import ClusteredL1PuppiConfig

from puppibuff import setup_from_config



def main():
    config = ClusteredL1PuppiConfig()
    config.n_events = 500_000

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
