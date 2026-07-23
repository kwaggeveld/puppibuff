from puppibuff.analyses.plotting import plot_distributions_flattened
from puppibuff.configs import FlatPuppiJetConfig

from puppibuff import setup_from_config, pt_power_weights


def main():
    config = FlatPuppiJetConfig()

    data, codec, model, x, y = setup_from_config(config)

    weights = pt_power_weights(data['pt'][:config.n_events], alpha = 0.3)
    model.fit(x, y, sample_weights = weights)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_distributions_flattened(data, samples, n_events = config.n_events)
    figure.show()
    input()


if __name__ == "__main__":
    main()
