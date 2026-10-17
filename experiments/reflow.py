from puppibuff import FlowBDT, load_model
from puppibuff.analyses import plot_histograms, sliced_wasserstein
from puppibuff.utils import flatten, initial_noise, output_dir

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

# 2-rectified flow (arXiv:2209.03003): deterministic noise-sample pairing to
# simplify the problem. Pair noise with where a trained flow takes it, then fit 
# a new flow on those pairs. The new paths are near straight and a few steps
# do trick. Dramatically decreases model size.

TEACHER   = "models/cond_const_relative_default"
N_STEPS   = 2                           # Of the reflowed grid model
N_PAIRS   = 1_000_000                   # N. of (noise, sample) pairs to reflow on
N_SAMPLES = 1_000_000                   # Plotting

CHANNELS  = [ "pt", "eta", "phi" ]

def n_bdts(flowbdt: FlowBDT) -> int:
    """BDTs the sampler evaluates: `ab2` never evaluates the last row, and
    slots that were never real have none.
    """
    return sum(bdt is not None for bdt in flowbdt.bdt_grid[:-1].flat)


def main():
    config, codec, flowbdt = load_model(TEACHER)
    data = config.dataset()

    reflowed = flowbdt.reflow(N_STEPS, N_PAIRS)

                                        # Same noise and conditions for both
    noise      = initial_noise((N_SAMPLES, flowbdt.n_channels))
    conditions = reflowed.sample_multiplicities(N_SAMPLES)
    teacher = codec.decode(flowbdt.sample(x0 = noise, conditions = conditions))
    samples = codec.decode(reflowed.sample(x0 = noise, conditions = conditions))

    target = flatten(data)
    print(f"SW1 flow   = { sliced_wasserstein(target, flatten(teacher), CHANNELS) :.4g}   "
          f"BDTs = { n_bdts(flowbdt) }")
    print(f"SW1 reflow = { sliced_wasserstein(target, flatten(samples), CHANNELS) :.4g}   "
          f"BDTs = { n_bdts(reflowed) }")

    figure = plot_histograms(data, { "Reflow": samples, "Flow matching": teacher })

    figure.savefig(output_dir(__file__) / f"reflow_s{ N_STEPS }.pdf")


if __name__ == "__main__":
    main()
