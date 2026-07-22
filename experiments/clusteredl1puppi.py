from puppibuff.analyses import plot_distributions
from puppibuff.configs import ClusteredL1PuppiConfig

from puppibuff import setup_from_config



def main():
    config = ClusteredL1PuppiConfig()
    config.n_events = 500_000

    data, codec, model, x, y = setup_from_config(config)

    print(data['pt'].shape)

    model.fit(x, y)

    raw_samples = model.sample(500_000)

    samples = codec.decode(raw_samples)

    figure = plot_distributions(
        data, samples, channels = ["pt", "eta", "phi"],
        n_events = config.n_events, mask_padding = True,
    )

    figure.show()
    input()


if __name__ == "__main__":
    main()
