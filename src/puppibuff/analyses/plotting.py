from __future__ import annotations

from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

#--- Shared utility functions ---

def _ratio(hist: NDArray, ref: NDArray) -> NDArray:
    """Compute histogram fraction `hist / ref` for nonzero `ref`s"""
    return np.divide(hist, ref, out = np.full_like(hist, np.nan), where = ref > 0)


def _make_grid(n_cols: int) -> tuple[Figure, NDArray]:
    """Return 2-row (main + ratio) * n_cols grid."""
    fig, axes = plt.subplots(
        2, n_cols, figsize = (5 * n_cols, 5.5), sharex = "col",
        gridspec_kw = { "height_ratios": [3, 1], "hspace": 0.05 },
    )
    return fig, np.reshape(axes, (2, n_cols))


def _finalise(fig: Figure, axes: NDArray, log_cols: list[int]) -> None:
    """Do final common actions on figure and axes"""
    for column in log_cols:
        axes[0, column].set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)


def _plot_channel(ax, rax, name: str, target: NDArray, sample: NDArray,
                   train: NDArray | None, bins: int) -> None:
    """One kinematic channel: target/sample/(optional train) density histograms
    on shared bin edges, plus a ratio panel.
    """
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
    edges   = np.histogram_bin_edges(np.concatenate([target, sample]), bins = bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_target, _ = np.histogram(target, bins = edges, density = True)
    hist_sample, _ = np.histogram(sample, bins = edges, density = True)

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

    if train is not None:               # Cut of dataset given?
        hist_train, _ = np.histogram(train, bins = edges, density = True)
                                        # Train distribution
        ax.hist(train, bins = edges, density = True, histtype = "step",
                color = "tab:green", alpha = 0.5, linewidth = 1.0,
                zorder = 2, label = "Train distribution")

                                        # Ratio sample / train
        rax.step(centers, _ratio(hist_sample, hist_train), where = "mid",
                 color = "tab:green", alpha = 0.7, linewidth = 1.0)

    ax.set_ylabel("Probability density")
    ax.grid(True, alpha = 0.3)
    ax.tick_params()

    rax.set_xlabel(name)
    rax.set_ylabel("Sample / reference")
    rax.grid(True, alpha = 0.3)


def _multiplicity(real: NDArray, n_events: int | None = None) -> NDArray:
    """Compute per-jet multiplicity as row-sum of the real mask"""
    return real[:n_events].sum(axis = 1).astype(int)


def _fractions(multiplicity: NDArray, length: int) -> NDArray:
    """Fraction of jets at each multiplicity 0 ... `length` - 1."""
    return np.bincount(multiplicity, minlength = length) / len(multiplicity)


def _plot_multiplicity(ax, rax, target_real: NDArray, sample_real: NDArray,
                        train_real: NDArray | None) -> None:
    """Create bar chart of target/sample/(optional train) jet-count
    fractions on a shared integer axis, plus a ratio panel.
    """
    target = _multiplicity(target_real)
    sample = _multiplicity(sample_real)

                                        # Compute shared bins and distribution
                                        # on these bins
    max_multiplicity  = max(target.max(), sample.max()) + 1
    bins = np.arange(max_multiplicity) 
    frac_target = _fractions(target, max_multiplicity)
    frac_sample = _fractions(sample, max_multiplicity)

                                        # Target distribution
    ax.bar(bins, frac_target, width = 1.0, color = "tab:gray", alpha = 0.25,
           edgecolor = "tab:gray", linewidth = 1.2, zorder = 1, label = "Target distribution")

                                        # Sample distribution
    ax.bar(bins, frac_sample, width = 1.0, fill = False, edgecolor = "tab:blue",
           linewidth = 1.8, zorder = 3, label = "Sample distribution")

                                        # Ratio sample / target
    rax.step(bins, _ratio(frac_sample, frac_target), where = "mid", color = "tab:blue", linewidth = 1.5)
    rax.axhline(1.0, color = "tab:gray", linestyle = "dashed", linewidth = 1.0, zorder = 0)

    if train_real is not None:          # Cut of dataset given?
        frac_train = _fractions(_multiplicity(train_real), max_multiplicity)

                                        # Train distribution
        ax.bar(bins, frac_train, width = 1.0, fill = False, edgecolor = "tab:green",
               alpha = 0.5, linewidth = 1.0, zorder = 2, label = "Train distribution")

                                        # Ratio sample / train
        rax.step(bins, _ratio(frac_sample, frac_train), where = "mid",
                 color = "tab:green", alpha = 0.7, linewidth = 1.0)

    ax.set_ylabel("Fraction of jets")
    ax.grid(True, alpha = 0.3)
    ax.tick_params()

    rax.set_xlabel("multiplicity")
    rax.set_ylabel("Sample / reference")
    rax.set_xticks(bins)                 # One tick per integer multiplicity
    rax.grid(True, alpha = 0.3)

#--- Public plotting functions ---

def plot_distributions_flattened(
    target: Dataset,                    # Truth channels (1D per channel)
    sample: dict[str, NDArray],         # Decoded, generated channels
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    bins: int = 75,
) -> Figure:
    """Flat (FlatPuppiJet-like) distributions: one column per channel."""
    channels = channels or target.channels()

    fig, axes = _make_grid(len(channels))
    for col, channel in enumerate(channels):
        train = target[channel][:n_events] if n_events is not None else None

        _plot_channel(axes[0, col], axes[1, col], channel,
                      target[channel], sample[channel], train, bins)

    _finalise(fig, axes, log_cols = [0])    # pt (first channel)
    return fig


def plot_distributions_jet(
    target: Dataset,                    # Kinematic channels + `real` mask
    sample: dict[str, NDArray],         # Decoded, same channels + `real`
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    bins: int = 75,
) -> Figure:
    """Fixed-multiplicity (ClusteredL1Puppi-like) distributions.
    Mask `sample["real"]` applied."""
    channels = channels or [ "pt", "eta", "phi" ]

    fig, axes = _make_grid(len(channels) + 1)    # + 1 for multiplicity column

    train_real = target["real"][:n_events] if n_events is not None else None

                                        # First panel: multiplicity distribution
    _plot_multiplicity(axes[0, 0], axes[1, 0], target["real"], sample["real"], train_real)

    target_mask = target["real"] > 0.5
    sample_mask = sample["real"] > 0.5

                                        # Remaining panels: feature distributions
    for col, channel in enumerate(channels, start = 1):
        train = None
        if n_events is not None:
            train_real = target["real"][:n_events] > 0.5
            train = target[channel][:n_events][train_real]

        _plot_channel(axes[0, col], axes[1, col], channel,
                      target[channel][target_mask],
                      sample[channel][sample_mask], train, bins)

    _finalise(fig, axes, log_cols = [0, 1])     # multiplicity, pt
    return fig
