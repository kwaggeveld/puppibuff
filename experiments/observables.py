from puppibuff import load_model
from puppibuff.utils import output_dir
from puppibuff.analyses import plot_histograms, pairwise_delta_r, axis_delta_r

import matplotlib.pyplot as plt
plt.style.use("puppibuff.style")

#-----------------------------------------------------------------------------

N_SAMPELS = 1_000_000
TRUNC     = 1_000_000

def main():
    config, codec, flowbdt = load_model("models/cond_const_relative_default")

    data = config.dataset()
    samples = codec.decode(flowbdt.sample(int(1e6)))

    samples_trunc = { channel: data[:TRUNC] for channel, data in samples.items() }

    observables = { "pairwise_delta_r": pairwise_delta_r, "axis_delta_r": axis_delta_r }
    figure = plot_histograms(
        target = { name: observable(data[:1_000_000]) for name, observable in observables.items() },
        samples = { 
            "Output": { name: observable(samples_trunc) for name, observable in observables.items() } 
        }
    )

    figure.savefig(output_dir(__file__) / f"histograms.pdf")


if __name__ == "__main__":
    main()
