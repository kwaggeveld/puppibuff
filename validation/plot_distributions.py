from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray


def plot_distributions(
    data: Dataset,
    samples: NDArray,
    channels: list[str] | None = None,
    bins: int = 100,
) -> plt.Figure:
    channels = channels or data.channels()

    fig, axes = plt.subplots(1, len(channels), figsize = (5 * len(channels), 4))
    axes = np.atleast_1d(axes)          # Keep axes iterable when len == 1
                                        
                                        # Convert array to dict to access with keys
                                        # TODO: factorise
    samples: dict[str, NDArray] = dict(zip(data.channels(), samples.T))

    for ax, channel in zip(axes, channels):
        real = np.asarray(data[channel]).ravel()
        gen  = np.asarray(samples[channel]).ravel()
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
        edges = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)

        ax.hist(real, bins = edges, density = True, histtype = "step", label = "data")
        ax.hist(gen,  bins = edges, density = True, histtype = "step", label = "generated")
        ax.set_xlabel(channel)
        ax.set_ylabel("density")
        ax.legend()

    fig.tight_layout()
    
    plt.show()

    return fig
