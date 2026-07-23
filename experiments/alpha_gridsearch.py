from puppibuff.analyses import total_mse, channel_mse
from puppibuff.analyses.plotting import plot_distributions_flattened
from puppibuff.configs import FlatPuppiJetConfig

from puppibuff import setup_from_config, pt_power_weights

import sys

#-----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} <alpha>")

    alpha = float(sys.argv[1])

    config = FlatPuppiJetConfig(n_events = None)    # Train on the entire dataset

    data, codec, model, x, y = setup_from_config(config)

    weights = pt_power_weights(data['pt'][:config.n_events], alpha = alpha)
    model.fit(x, y, sample_weights = weights)

    raw_samples = model.sample(500_000)
    samples = codec.decode(raw_samples)

    loss = total_mse(data, samples)
    pt_loss = channel_mse(data['pt'], samples['pt'])
    print(f"alpha = {alpha:g}  ->  total_mse = {loss:.6g}  pt_mse = {pt_loss:.6g}")

    figure = plot_distributions_flattened(data, samples, n_events = config.n_events)
    figure.savefig(f"alpha_gridsearch_alpha{alpha:g}.pdf", format = "pdf")


if __name__ == "__main__":
    main()
