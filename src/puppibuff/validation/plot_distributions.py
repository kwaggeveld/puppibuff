from __future__ import annotations

from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from matplotlib.figure import Figure


def plot_distributions(
    data: Dataset,
    samples: dict[str, NDArray],        # decoded, physical-space channels
    channels: list[str] | None = None,
    bins: int = 100,
) -> Figure:
    channels = channels or data.channels()

    fig, axes = plt.subplots(1, len(channels), figsize = (5 * len(channels), 4))
    axes = np.atleast_1d(axes)          # Keep axes iterable when len == 1

    for ax, channel in zip(axes, channels):
        real = np.asarray(data[channel]).ravel()
        gen  = np.asarray(samples[channel]).ravel()
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
        edges = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)

        ax.hist(real, bins = edges, density = True, histtype = "step", label = "Target distribution")
        ax.hist(gen,  bins = edges, density = True, histtype = "step", label = "Sample distribution")
        ax.set_xlabel(channel)
        ax.set_ylabel("Probability density")
        ax.legend()

    fig.tight_layout()

    return fig
