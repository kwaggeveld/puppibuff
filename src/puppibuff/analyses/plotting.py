from __future__ import annotations

from ..datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from typing import Callable
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


                                        # Per-series styling. The colour is kept
                                        # separate because an outline is `color =`
                                        # for a line (hist/KDE) but `edgecolor=`
                                        # for a bar.
_TARGET_C, _SAMPLE_C, _TRAIN_C = "tab:gray", "tab:blue", "tab:green"

_TARGET = dict(color = _TARGET_C, edgecolor = _TARGET_C, alpha = 0.25,
               linewidth = 1.2, zorder = 1, label = "Target distribution")
_SAMPLE = dict(linewidth = 1.8, zorder = 3, label = "Sample distribution")
_TRAIN  = dict(alpha = 0.5, linewidth = 1.0, zorder = 2, label = "Train distribution")


def _plot_ratio(rax, x: NDArray, sample_over_target: NDArray,
                training_cut: NDArray | None, step: bool) -> None:
    """Ratio panel: sample/target (and sample/train, if given) plus the unit
    line. `step` picks stepped (histogram/bar) over smooth (KDE) rendering.
    """
    draw    = rax.step if step else rax.plot
    step_kw = { "where": "mid" } if step else {}

    draw(x, sample_over_target, color = _SAMPLE_C, linewidth = 1.5, **step_kw)
    rax.axhline(1.0, color = _TARGET_C, linestyle = "dashed", linewidth = 1.0, zorder = 0)

    if training_cut is not None:
        draw(x, training_cut, color = _TRAIN_C, alpha = 0.7, linewidth = 1.0, **step_kw)


def _finish_panel(ax, rax, xlabel: str, ylabel: str) -> None:
    """Shared axis labels and grid for one main = ratio column."""
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha = 0.3)
    ax.tick_params()

    rax.set_xlabel(xlabel)
    rax.set_ylabel("Sample / reference")
    rax.grid(True, alpha = 0.3)


def _plot_channel(ax, rax, name: str, target: NDArray, sample: NDArray,
                   training_cut: NDArray | None, bins: int) -> None:
    """One kinematic channel: target/sample/(optional train) density histograms
    on shared bin edges, plus a ratio panel.
    """
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
    edges   = np.histogram_bin_edges(np.concatenate([target, sample]), bins = bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_target, _, _ = ax.hist(target, bins = edges, density = True,
                                histtype = "stepfilled", **_TARGET)
    hist_sample, _, _ = ax.hist(sample, bins = edges, density = True,
                                histtype = "step", color = _SAMPLE_C, **_SAMPLE)

    ratio_train = None
    if training_cut is not None:               # Cut of dataset given?
        hist_train, _, _ = ax.hist(training_cut, bins = edges, density = True,
                                   histtype = "step", color = _TRAIN_C, **_TRAIN)
        ratio_train = _ratio(hist_sample, hist_train)

    _plot_ratio(rax, centers, _ratio(hist_sample, hist_target), ratio_train, step = True)
    _finish_panel(ax, rax, name, "Probability density")


_KDE_MAX_POINTS = 50_000                # A KDE eval is O(n_data * len(grid));
                                        # 50k points already give a smooth
                                        # estimate, so cap to stay responsive

def _kde(data: NDArray) -> gaussian_kde:
    """Return Gaussian-KDE estimator for `data`, subsampling large inputs."""
    if len(data) > _KDE_MAX_POINTS:     # Deterministic subsample => stable plots
        data = data[np.random.default_rng(0).choice(len(data), _KDE_MAX_POINTS, replace = False)]

    return gaussian_kde(data)


def _density_ratio(sample: NDArray, ref: NDArray) -> NDArray:
    """`sample / ref`, blanked where the reference KDE drops below 0.1% of its
    peak: its exp-decaying tails reach ~0 far from any real support and would
    otherwise send the ratio to +/-inf (and overflow the divide).
    """
    return _ratio(sample, np.where(ref < ref.max() * 1e-3, 0.0, ref))


def _plot_channel_density(ax, rax, name: str, target: NDArray, sample: NDArray,
                          train: NDArray | None, points: int) -> None:
    """One kinematic channel: target/sample/(optional train) Gaussian-KDE
    densities on a shared evaluation grid, plus a ratio panel. The smooth-curve
    counterpart of `_plot_channel`.
    """
                                        # Shared grid so the curves and their
                                        # ratio are directly comparable
    grid = np.linspace(min(target.min(), sample.min()),
                       max(target.max(), sample.max()), points)

    dens_target = _kde(target)(grid)
    dens_sample = _kde(sample)(grid)

    ax.fill_between(grid, dens_target, **_TARGET)
    ax.plot(grid, dens_sample, color = _SAMPLE_C, **_SAMPLE)

    ratio_train = None
    if train is not None:               # Cut of dataset given?
        dens_train = _kde(train)(grid)
        ax.plot(grid, dens_train, color = _TRAIN_C, **_TRAIN)
        ratio_train = _density_ratio(dens_sample, dens_train)

    _plot_ratio(rax, grid, _density_ratio(dens_sample, dens_target), ratio_train, step = False)
    _finish_panel(ax, rax, name, "Probability density")


def _multiplicity(real: NDArray) -> NDArray:
    """Per-jet multiplicity as the row-sum of the real mask."""
    return real.sum(axis = 1).astype(int)


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

    length = max(target.max(), sample.max()) + 1     # Shared integer bins
    bins   = np.arange(length)
    frac_target = _fractions(target, length)
    frac_sample = _fractions(sample, length)

    ax.bar(bins, frac_target, width = 1.0, **_TARGET)
    ax.bar(bins, frac_sample, width = 1.0, fill = False, edgecolor = _SAMPLE_C, **_SAMPLE)

    ratio_train = None
    if train_real is not None:          # Cut of dataset given?
        frac_train = _fractions(_multiplicity(train_real), length)
        ax.bar(bins, frac_train, width = 1.0, fill = False, edgecolor = _TRAIN_C, **_TRAIN)
        ratio_train = _ratio(frac_sample, frac_train)

    _plot_ratio(rax, bins, _ratio(frac_sample, frac_target), ratio_train, step = True)
    _finish_panel(ax, rax, "multiplicity", "Fraction of jets")
    rax.set_xticks(bins)                 # One tick per integer multiplicity


def _plot_grid(target: Dataset, sample: dict[str, NDArray],
               channels: list[str] | None, n_events: int | None,
               plot_channel: Callable, resolution: int) -> tuple[Figure, NDArray, list[int], int]:
    """Shared flat/jet iteration behind `plot_histograms`/`plot_distributions`. Detect
    jet vs flat on `"real" in sample`, build the column grid (a leading
    multiplicity bar panel for jet data), drop padding on both sides, and call
    `plot_channel(ax, rax, name, truth, sampled, train, resolution)` per channel,
    where `resolution` is the histogram bin count or the KDE grid size. Return
    `(fig, axes, log_cols, offset)`, where `offset` is the leading multiplicity 
    column count (1 for jets, else 0).
    """
    jet      = "real" in sample
    channels = channels or (["pt", "eta", "phi"] if jet else target.channels())
    offset   = 1 if jet else 0          # Leading multiplicity column for jets

    fig, axes = _make_grid(len(channels) + offset)

                                        # Flat data keeps every element, so its
                                        # masks are the no-op `slice(None)` (i.e.
                                        # `arr[:]`); a jet instead drops padding
                                        # via its `real` mask.
    target_mask: slice | NDArray = slice(None)
    sample_mask: slice | NDArray = slice(None)
    train_mask:  slice | NDArray = slice(None)

    if jet:                             # First panel: multiplicity distribution
        train_real = target["real"][:n_events] if n_events is not None else None
        _plot_multiplicity(axes[0, 0], axes[1, 0], target["real"], sample["real"], train_real)

        target_mask = target["real"] > 0.5
        sample_mask = sample["real"] > 0.5
        if n_events is not None:
            train_mask = target["real"][:n_events] > 0.5

    for column, channel in enumerate(channels, start = offset):
        truth   = target[channel][target_mask]
        sampled = sample[channel][sample_mask]
        train   = target[channel][:n_events][train_mask] if n_events is not None else None

        plot_channel(axes[0, column], axes[1, column], channel, truth, sampled, train, resolution)

    return fig, axes, [0, 1] if jet else [0], offset

#--- Public plotting functions ---

def plot_histograms(
    target: Dataset,                    # Truth channels (+ `real` for jets)
    sample: dict[str, NDArray],         # Decoded, generated channels
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    bins: int = 75,
) -> Figure:
    """Binned target/sample/(optional train) distributions with ratio panels.
    Dispatches on `"real" in sample`: padded jet data gets a leading
    multiplicity bar panel and has its padding masked off; flat data gets one
    column per channel.
    """
    fig, axes, log_cols, _ = _plot_grid(target, sample, channels, n_events, _plot_channel, bins)
    _finalise(fig, axes, log_cols)
    return fig


def plot_distributions(
    target: Dataset,                    # Truth channels (+ `real` for jets)
    sample: dict[str, NDArray],         # Decoded, generated channels
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    points: int = 200,                  # KDE evaluation-grid resolution
) -> Figure:
    """Kernel-density-estimate distributions: the smooth-curve counterpart of
    `plot_histograms`, approximating each channel's PDF with a Gaussian KDE. 
    """
    fig, axes, log_cols, offset = _plot_grid(target, sample, channels, n_events,
                                             _plot_channel_density, points)
    _finalise(fig, axes, log_cols)

                                        # A KDE never reaches 0, so its tails
                                        # break a log axis' autoscale.
                                        # Pin each log-scaled density
                                        # panel from its peak down to 8 orders
                                        # below.
    for col in log_cols:
        if col >= offset:               # A KDE panel, not the multiplicity bars
            ax   = axes[0, col]
            peak = max(line.get_ydata().max() for line in ax.lines)
            ax.set_ylim(peak * 1e-8, peak * 2)

    return fig
