from puppibuff import build_trainds, FlowBDT
from puppibuff.analyses import plot_histograms, sliced_wasserstein
from puppibuff.configs import FlatPuppiJetConfig
from puppibuff.utils import initial_noise, output_dir

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

# 2-rectified flow (arXiv:2209.03003): deterministic noise-sample pairing to
# simplify the problem. Pair noise with where a trained flow takes it, then fit 
# a new flow on those pairs. The new paths are near straight and a few steps
# do trick. Dramatically decreases model size.

N_STEPS = 2                             # Of the reflowed grid model

def main():
    config = FlatPuppiJetConfig()

    data, codec, flow, x, y = config.setup()
    flow.fit(x, y)

    x0 = initial_noise((len(x.x0), flow.n_channels))

    reflow = FlowBDT(config.tree_config)
    reflow.fit(*build_trainds(flow.sample(x0 = x0), N_STEPS, x0))

    noise   = initial_noise((1_000_000, flow.n_channels))
    teacher = codec.decode(flow.sample(x0 = noise))
    samples = codec.decode(reflow.sample(x0 = noise))
    
    
                                        # `ab2`` never evaluates BDTs on the 
                                        # the last row, so not included in count
    print(f"SW1 flow   = { sliced_wasserstein(data, teacher) :.4g}   "
          f"BDTs = { flow.bdt_grid[:-1].size }")
    print(f"SW1 reflow = { sliced_wasserstein(data, samples) :.4g}   "
          f"BDTs = { reflow.bdt_grid[:-1].size }")

    figure = plot_histograms(
        data, samples, overlay = teacher,
        labels = { "Output": "Reflow", "Training": "Flow matching" },
    )

    figure.savefig(output_dir(__file__) / f"reflow_s{ N_STEPS }.pdf")


if __name__ == "__main__":
    main()
