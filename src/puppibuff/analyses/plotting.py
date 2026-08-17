from __future__ import annotations

from ..datasets import Dataset

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


_KDE_MAX_POINTS = 50_000                # A KDE eval is O(n_data * len(grid))

def _kde(data: NDArray) -> gaussian_kde:
    """Return Gaussian KDE estimator for `data`, subsampling large inputs.
    `data` is 1D for a marginal, or `(n_channels, n_events)` for a joint
    density. NB: transposed compared to normal.
    """
    n_points = data.shape[-1]
    if n_points > _KDE_MAX_POINTS:
        data = data[..., np.random.default_rng(0).choice(n_points, _KDE_MAX_POINTS, replace = False)]

    return gaussian_kde(data)


def _mass_levels(density: NDArray, quantiles: tuple[float, ...]) -> NDArray:
    """Density levels enclosing the given fractions of the total probability
    mass, ascending as `contour` requires. Sorting descending turns the
    cumulative sum into "mass at or above this level", so a quantile's level is
    where that first reaches it. `np.unique` both sorts and drops the duplicates
    two quantiles produce on a flat density, which `contour` rejects.
    """
    ranked   = np.sort(density, axis = None)[::-1]
    enclosed = np.cumsum(ranked) / ranked.sum()

    return np.unique(ranked[np.clip(np.searchsorted(enclosed, quantiles), 0, ranked.size - 1)])


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


_QUANTILES = (.25, .5, .75, .95)        # Contours enclose these fractions of the
                                        # total probability mass

_SPAN = (.1, 99.9)                      # Percentiles the evaluation grid spans.
                                        # Excluding extreme percentiles removes
                                        # outliers.

def _joint_density(marginals: tuple[NDArray, ...], grid: NDArray) -> NDArray:
    """Joint KDE of `marginals` evaluated over `grid`, a `(n_axes, *shape)` 
    meshgrid, and reshaped back to that grid.
    """
    return _kde(np.stack(marginals))(grid.reshape(len(marginals), -1)).reshape(grid.shape[1:])


def _plot_contour(ax, grid: NDArray, density: NDArray, color: str, style: str,
                  width: float, zorder: int, fill: bool) -> None:
    """One combination's mass contours."""
    levels = _mass_levels(density, _QUANTILES)

    if fill:
        ax.contourf(*grid, density, levels = levels, colors = color,
                    alpha = .25, zorder = zorder, extend = "max")

    ax.contour(*grid, density, levels = levels, colors = color,
               linestyles = style, linewidths = width, zorder = zorder + 1)


_LOG_CHANNELS = ( "pt" )                # `pt` spans orders of magnitude while
                                        # eta/phi are O(1)

def _rescale(names: tuple[str, str], series: tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
    """Map one point cloud into the space its KDE is estimated in. Take logarithm
    of channels in `_LOG_CHANNELS`, otherwise identity map.
    """
    first, second = ( np.log1p(values) if channel in _LOG_CHANNELS else values
                      for channel, values in zip(names, series) )

    return first, second


def _axis_label(name: str) -> str:
    """Name the plotted variable, which is the rescaled one for `_LOG_CHANNELS`."""
    return f"log1p({ name })" if name in _LOG_CHANNELS else name


def _plot_pair_density(
    ax, 
    pair: tuple[str, str], 
    target: tuple[NDArray, NDArray],
    sample: tuple[NDArray, NDArray],
    train: tuple[NDArray, NDArray] | None, 
    points: int
) -> None:
    """One channel pair: target/sample/(optional train) joint-KDE contours on a
    shared evaluation grid. The 2-D counterpart of `_plot_channel_density`.
    """
    target = _rescale(pair, target)
    sample = _rescale(pair, sample)
    train  = train if train is None else _rescale(pair, train)

                                        # Shared grid so the contour sets are
    grids = []                          # directly comparable
    for truth, sampled in zip(target, sample):
        low, high = np.percentile(np.concatenate([truth, sampled]), _SPAN)
        grids.append(np.linspace(low, high, points))

    grid = np.stack(np.meshgrid(grids[0], grids[1], indexing = "ij"))

                                        # Target
    _plot_contour(ax, grid, _joint_density(target, grid), _TARGET_C, "solid",
                  width = 0.8, zorder = 1, fill = True)

    if train is not None:               # Training
        _plot_contour(ax, grid, _joint_density(train, grid), _TRAIN_C, "dotted",
                      width = 0.5, zorder = 3, fill = False)

                                        # Sampled
    _plot_contour(ax, grid, _joint_density(sample, grid), _SAMPLE_C, "dashed",
                  width = 1, zorder = 5, fill = False)

    ax.set_xlabel(_axis_label(pair[0]))
    ax.set_ylabel(_axis_label(pair[1]))
    ax.grid(True, alpha = 0.3)


def _channel_data(
    target: Dataset, 
    sample: dict[str, NDArray],
    channels: list[str] | None, 
    n_events: int | None,
) -> dict[str, tuple[NDArray, NDArray, NDArray | None]]:
    """Construct per-channel `(truth, sampled, train)` arrays, with jet padding 
    dropped.
    """
    jet      = "real" in sample
    channels = channels or (["pt", "eta", "phi"] if jet else target.channels())

                                        # Flat data keeps every element, so its
                                        # masks are the no-op `slice(None)` (i.e.
                                        # `arr[:]`); a jet instead drops padding
                                        # via its `real` mask.
    target_mask: slice | NDArray = slice(None)
    sample_mask: slice | NDArray = slice(None)
    train_mask:  slice | NDArray = slice(None)

    if jet:
        target_mask = target["real"] > 0.5
        sample_mask = sample["real"] > 0.5
        if n_events is not None:
            train_mask = target["real"][:n_events] > 0.5

    return { channel: (target[channel][target_mask],
                       sample[channel][sample_mask],
                       target[channel][:n_events][train_mask] if n_events is not None else None)
             for channel in channels }


def _plot_grid(target: Dataset, sample: dict[str, NDArray],
               channels: list[str] | None, n_events: int | None,
               plot_channel: Callable, resolution: int) -> tuple[Figure, NDArray, list[int], int]:
    """Shared flat/jet iteration behind `plot_histograms`/`plot_distributions`.
    Build the column grid (a leading multiplicity bar panel for jet data) and
    call `plot_channel(ax, rax, name, truth, sampled, train, resolution)` per
    channel, where `resolution` is the histogram bin count or the KDE grid size.
    Return `(fig, axes, log_cols, offset)`, where `offset` is the leading
    multiplicity column count (1 for jets, else 0).
    """
    jet    = "real" in sample
    data   = _channel_data(target, sample, channels, n_events)
    offset = 1 if jet else 0            # Leading multiplicity column for jets

    fig, axes = _make_grid(len(data) + offset)

    if jet:                             # First panel: multiplicity distribution
        train_real = target["real"][:n_events] if n_events is not None else None
        _plot_multiplicity(axes[0, 0], axes[1, 0], target["real"], sample["real"], train_real)

    for column, (channel, series) in enumerate(data.items(), start = offset):
        plot_channel(axes[0, column], axes[1, column], channel, *series, resolution)

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


def plot_contours(
    target: Dataset,                    # Truth channels (+ `real` for jets)
    sample: dict[str, NDArray],         # Decoded, generated channels
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    points: int = 100,                  # KDE evaluation-grid resolution per axis
) -> Figure:
    """Make pairwise joint-KDE contour plots. One panel per channel pair, each
    contour enclosing a fixed fraction of the joined probability mass.
    """
    data  = _channel_data(target, sample, channels, n_events)
    pairs = list(combinations(data, 2))

    fig, axes = plt.subplots(1, len(pairs), figsize = (5 * len(pairs), 5), squeeze = False)

    for ax, pair in zip(axes[0], pairs):
        truth, sampled, train = zip(*(data[name] for name in pair))

        _plot_pair_density(ax, pair, truth, sampled,
                   None if train[0] is None else train, points)

                                        # A contour registers no legend handle,
                                        # so the labels ride on empty proxies.
    handles = [ Line2D([], [], color = _TARGET_C, linestyle = "solid",  label = _TARGET["label"]),
                Line2D([], [], color = _SAMPLE_C, linestyle = "dashed", label = _SAMPLE["label"]) ]

    if n_events is not None:            # Cut of dataset given?
        handles.append(Line2D([], [], color = _TRAIN_C, linestyle = "dotted", label = _TRAIN["label"]))

    fig.legend(handles = handles)

    return fig
