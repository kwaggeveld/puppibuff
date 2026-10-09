from puppibuff.analyses import plot_histograms, sliced_wasserstein
from puppibuff.build_trainds import Paths
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.utils import initial_noise, output_dir

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

# Attempting to one-shot the problem by regressing data straight from noise.
# ...Doesn't work: noise and data are drawn independently, so E[x1 | x0] = E[x1] 
# and the predictions just collapse to the per-channel mean...

def main():
    config = FlatPuppiJetConfig()

    data, codec, model, x, _ = config.setup()

    model.fit(Paths(x.x0, x.x1, np.array([ 0 ])), x.x1)

    raw_samples = model.predict(0., initial_noise((1_000_000, model.n_channels)))

    samples = codec.decode(raw_samples)
    print(f"SW1 = { sliced_wasserstein(data, samples) :.4g}")

    figure = plot_histograms(data, { "Output": samples })

    figure.savefig(output_dir(__file__) / "direct_regression.pdf")


if __name__ == "__main__":
    main()
