from __future__ import annotations

from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

def plot_distributions(
    data: Dataset,
    samples: dict[str, NDArray],        # decoded, physical-space channels
    channels: list[str] | None = None,
    n_events: int | None = None,
    bins: int = 75,
) -> Figure:
    channels = channels or data.channels()
    n_cols = len(channels)

    fig, axes = plt.subplots(
        2, n_cols, figsize = (5 * n_cols, 5.5), sharex = "col",
        gridspec_kw = { "height_ratios": [3, 1], "hspace": 0.05 },
    )
    axes = np.reshape(axes, (2, n_cols)) # plt.subplots squeezes to 1D when n_cols == 1

    for col, channel in enumerate(channels):
        ax, rax = axes[0, col], axes[1, col]

        real  = np.asarray(data[channel]).ravel()
        gen   = np.asarray(samples[channel]).ravel()
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
        edges   = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        hist_real, _ = np.histogram(real, bins = edges, density = True)
        hist_gen,  _ = np.histogram(gen,  bins = edges, density = True)

                                        # Target distribution
        ax.hist(real, bins = edges, density = True, histtype = "stepfilled",
                color = "tab:gray", alpha = 0.25, edgecolor = "tab:gray",
                linewidth = 1.2, zorder = 1, label = "Target distribution")

                                        # Sample distribution
        ax.hist(gen, bins = edges, density = True, histtype = "step",
                color = "tab:blue", linewidth = 1.8, zorder = 3, label = "Sample distribution")

        ratio_gen = np.divide(hist_gen, hist_real,
                               out = np.full_like(hist_gen, np.nan), where = hist_real > 0)
        rax.step(centers, ratio_gen, where = "mid", color = "tab:blue", linewidth = 1.5)
        rax.axhline(1.0, color = "tab:gray", linestyle = "dashed", linewidth = 1.0, zorder = 0)


        if n_events is not None:        # Cut of dataset given?
            train = np.asarray(data[channel][:n_events]).ravel()
            hist_train, _ = np.histogram(train, bins = edges, density = True)
                                        # Train distribution
            ax.hist(train, bins = edges, density = True, histtype = "step",
                    color = "tab:green", alpha = 0.5, linewidth = 1.0,
                    zorder = 2, label = "Train distribution")

            ratio_train = np.divide(hist_train, hist_real,
                                    out = np.full_like(hist_train, np.nan), where = hist_real > 0)
            rax.step(centers, ratio_train, where = "mid",
                     color = "tab:green", alpha = 0.7, linewidth = 1.0)


        ax.set_ylabel("Probability density")
        ax.grid(True, alpha = 0.3)
        ax.tick_params(labelbottom = False)

        rax.set_xlabel(channel)
        rax.set_ylabel("Ratio / Target")
        rax.grid(True, alpha = 0.3)

    axes[0, 0].set_yscale("log")        # pt to log

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)
    
    fig.tight_layout()

    return fig
