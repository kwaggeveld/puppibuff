from puppibuff.analyses import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig


def main():
    config = FlatPuppiJetConfig()

    data, codec, model, x, y = config.setup()

    model.fit(x, y)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_histograms(
        data, samples, n_events = config.n_events,
    )

    figure.savefig("figures/puppijet_full_d6_n50.pdf")


if __name__ == "__main__":
    main()
