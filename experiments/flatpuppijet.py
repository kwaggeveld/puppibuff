from puppibuff.analyses.plotting import plot_distributions_flattened
from puppibuff.configs import FlatPuppiJetConfig

from puppibuff import setup_from_config



def main():
    config = FlatPuppiJetConfig(n_events = None)

    data, codec, model, x, y = setup_from_config(config)

    model.fit(x, y)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_distributions_flattened(
        data, samples, channels = ["pt", "eta", "phi"], n_events = config.n_events,
    )

    figure.savefig("figures/puppijet_full_d6_n50.pdf")


if __name__ == "__main__":
    main()
