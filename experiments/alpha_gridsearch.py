from puppibuff.analyses import plot_histograms
from puppibuff.configs import FlatPuppiJetConfig

from puppibuff.utils import output_dir
from puppibuff.weighting import pt_power_weights

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

ALPHAS = [ 0, 2, 4 ]

def main():
    outdir = output_dir(__file__)

    config = FlatPuppiJetConfig()

    data, codec, model, x, y = config.setup()

    for alpha in ALPHAS:
        weights = pt_power_weights(data['pt'][:config.n_events], alpha = alpha)
        model.fit(x, y, sample_weights = weights)

        raw_samples = model.sample(500_000)
        samples = codec.decode(raw_samples)

        figure = plot_histograms(data, samples, channels = ["pt"], width = 6.3 * 0.3)
        figure.savefig(outdir / f"alpha{alpha:g}.pdf", format = "pdf")


if __name__ == "__main__":
    main()
