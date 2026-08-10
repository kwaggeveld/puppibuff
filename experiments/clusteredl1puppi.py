from puppibuff.analyses.plotting import plot_histograms
from puppibuff.configs import ClusteredL1PuppiConfig


def main():
    config = ClusteredL1PuppiConfig()
    config.n_events = 500_000

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
