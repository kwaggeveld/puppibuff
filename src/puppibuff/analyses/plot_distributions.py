from __future__ import annotations

from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

def _masked_values(source, channel: str, mask_padding: bool, n_events: int | None = None) -> NDArray:
    values = np.asarray(source[channel][:n_events]).ravel()

    if mask_padding:                    # Drop padded slots via the `real` flag
        values = values[np.asarray(source["real"][:n_events]).ravel() > 0.5]

    return values


def _ratio(hist: NDArray, ref: NDArray) -> NDArray:
    return np.divide(hist, ref, out = np.full_like(hist, np.nan), where = ref > 0)


def _plot_channel(ax, rax, channel: str, data: Dataset, samples: dict[str, NDArray],
                   n_events: int | None, mask_padding: bool, bins: int) -> None:
    
    target = _masked_values(data,    channel, mask_padding)
    sample = _masked_values(samples, channel, mask_padding)
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
    edges   = np.histogram_bin_edges(np.concatenate([target, sample]), bins = bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_target, _ = np.histogram(target, bins = edges, density = True)
    hist_sample,  _ = np.histogram(sample,  bins = edges, density = True)

                                        # Target distribution
    ax.hist(target, bins = edges, density = True, histtype = "stepfilled",
            color = "tab:gray", alpha = 0.25, edgecolor = "tab:gray",
            linewidth = 1.2, zorder = 1, label = "Target distribution")

                                        # Sample distribution
    ax.hist(sample, bins = edges, density = True, histtype = "step",
            color = "tab:blue", linewidth = 1.8, zorder = 3, label = "Sample distribution")

                                        # Ratio sample / target
    rax.step(centers, _ratio(hist_sample, hist_target), where = "mid", color = "tab:blue", linewidth = 1.5)
    rax.axhline(1.0, color = "tab:gray", linestyle = "dashed", linewidth = 1.0, zorder = 0)

    if n_events is not None:            # Cut of dataset given?
        train = _masked_values(data, channel, mask_padding, n_events)
        hist_train, _ = np.histogram(train, bins = edges, density = True)
                                        # Train distribution
        ax.hist(train, bins = edges, density = True, histtype = "step",
                color = "tab:green", alpha = 0.5, linewidth = 1.0,
                zorder = 2, label = "Train distribution")

        rax.step(centers, _ratio(hist_train, hist_target), where = "mid",
                 color = "tab:green", alpha = 0.7, linewidth = 1.0)

    ax.set_ylabel("Probability density")
    ax.grid(True, alpha = 0.3)
    ax.tick_params(labelbottom = False)

    rax.set_xlabel(channel)
    rax.set_ylabel("Ratio / Target")
    rax.grid(True, alpha = 0.3)


def plot_distributions(
    data: Dataset,
    samples: dict[str, NDArray],        # Decoded, physical-space channels
    channels: list[str] | None = None,
    n_events: int | None = None,
    bins: int = 75,
    mask_padding: bool = False,         # Keep only real particles
) -> Figure:
    channels = channels or data.channels()
    n_cols = len(channels)

    fig, axes = plt.subplots(
        2, n_cols, figsize = (5 * n_cols, 5.5), sharex = "col",
        gridspec_kw = { "height_ratios": [3, 1], "hspace": 0.05 },
    )
    axes = np.reshape(axes, (2, n_cols)) # plt.subplots squeezes to 1D when n_cols == 1

    for col, channel in enumerate(channels):
        _plot_channel(axes[0, col], axes[1, col], channel, data, samples, n_events, mask_padding, bins)

    axes[0, 0].set_yscale("log")        # pt to log

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

    fig.tight_layout()

    return fig
