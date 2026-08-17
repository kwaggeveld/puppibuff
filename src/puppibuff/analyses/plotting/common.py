from __future__ import annotations

from ...datasets import Dataset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from typing import Callable
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

Events = dict[str, NDArray]             # One distribution's per-channel arrays

#--- Styling defaults ---

                                        # The colour kept separate because keyw.
                                        # differs between line/bar
TARGET_C, SAMPLE_C, TRAIN_C = "tab:gray", "tab:blue", "tab:green"

TARGET: dict = dict(color = TARGET_C, edgecolor = TARGET_C, alpha = 0.25,
                    linewidth = 1.2, zorder = 1, label = "Target distribution")
SAMPLE: dict = dict(linewidth = 1.8, zorder = 3, label = "Sample distribution")
TRAIN:  dict = dict(alpha = 0.5, linewidth = 1.0, zorder = 2, label = "Train distribution")

TARGET_CONTOUR: dict = dict(colors = TARGET_C, linestyles = "solid",  linewidths = 0.8, zorder = 1)
TRAIN_CONTOUR:  dict = dict(colors = TRAIN_C,  linestyles = "dotted", linewidths = 0.5, zorder = 3)
SAMPLE_CONTOUR: dict = dict(colors = SAMPLE_C, linestyles = "dashed", linewidths = 1.0, zorder = 5)

LOG_CHANNELS = set("pt")                # `pt` spans orders of magnitude while
                                        # eta/phi are O(1). Jets do as well.
LOG_COLUMNS  = LOG_CHANNELS | set("multiplicity")

#--- Numeric helpers ---

def ratio(hist: NDArray, ref: NDArray) -> NDArray:
    """Compute histogram fraction `hist / ref` for nonzero `ref`s"""
    return np.divide(hist, ref, out = np.full_like(hist, np.nan), where = ref > 0)


KDE_MAX_POINTS = 50_000                 # A KDE eval is O(n_data * len(grid))

def kde(data: NDArray) -> gaussian_kde:
    """Return Gaussian KDE estimator for `data`, subsampling large inputs.
    `data` is 1D for a marginal, or `(n_channels, n_events)` for a joint
    density. NB: transposed compared to normal.
    """
    n_points = data.shape[-1]
    if n_points > KDE_MAX_POINTS:
        data = data[..., np.random.default_rng(0).choice(n_points, KDE_MAX_POINTS, replace = False)]

    return gaussian_kde(data)


def _unpad(source: Dataset | dict[str, NDArray], channels: list[str],
           real: NDArray | None) -> Events:
    """Per-channel arrays for one series. Flat data keeps every element, so it
    passes `real = None` for the no-op `arr[:]`; a jet drops its padding.
    """
    mask = slice(None) if real is None else real > 0.5

    return { channel: source[channel][mask] for channel in channels }


def channel_data(
    target: Dataset,
    sample: dict[str, NDArray],
    channels: list[str] | None,
    n_events: int | None,
) -> tuple[Events, Events, Events | None]:
    """Construct the `(target, sample, train)` arrays every plotter draws, with
    jet padding dropped. `train` is the trained-on cut of `target`, or `None`
    when no `n_events` is given.
    """
    jet      = "real" in sample
    channels = channels or [ channel for channel in target.channels() if channel != "real" ]

                                        # A `Dataset` slices into a `Dataset`, so
                                        # the train series is the target one over
                                        # a shorter dataset.
    train = None if n_events is None else target[:n_events]

    return (_unpad(target, channels, target["real"] if jet else None),
            _unpad(sample, channels, sample["real"] if jet else None),
            None if train is None else _unpad(train, channels, train["real"] if jet else None))

#--- Panel scaffolding ---

def make_grid(n_cols: int) -> tuple[Figure, NDArray]:
    """Return 2-row (main + ratio) * n_cols grid."""
    fig, axes = plt.subplots(
        2, n_cols, figsize = (5 * n_cols, 5.5), sharex = "col",
        gridspec_kw = { "height_ratios": [3, 1], "hspace": 0.05 },
    )
    return fig, np.reshape(axes, (2, n_cols))


def plot_ratio(rax: Axes, x: NDArray, sample_over_target: NDArray,
               training_cut: NDArray | None, step: bool) -> None:
    """Ratio panel: sample/target (and sample/train, if given) plus the unit
    line. `step` picks stepped (histogram/bar) over smooth (KDE) rendering.
    """
    draw = rax.step if step else rax.plot

    step_kw: dict = { "where": "mid" } if step else {}

    draw(x, sample_over_target, color = SAMPLE_C, linewidth = 1.5, **step_kw)
    rax.axhline(1.0, color = TARGET_C, linestyle = "dashed", linewidth = 1.0, zorder = 0)

    if training_cut is not None:
        draw(x, training_cut, color = TRAIN_C, alpha = 0.7, linewidth = 1.0, **step_kw)


def finish_panel(ax: Axes, rax: Axes, xlabel: str, ylabel: str) -> None:
    """Shared axis labels and grid for one main = ratio column."""
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha = 0.3)

    rax.set_xlabel(xlabel)
    rax.set_ylabel("Sample / reference")
    rax.grid(True, alpha = 0.3)


def _multiplicity(real: NDArray) -> NDArray:
    """Per-jet multiplicity as the row-sum of the real mask."""
    return real.sum(axis = 1).astype(int)


def _fractions(multiplicity: NDArray, length: int) -> NDArray:
    """Fraction of jets at each multiplicity 0 ... `length` - 1."""
    return np.bincount(multiplicity, minlength = length) / len(multiplicity)


def multiplicity_panel(ax: Axes, rax: Axes, target_real: NDArray,
                       sample_real: NDArray, train_real: NDArray | None) -> None:
    """Create bar chart of target/sample/(optional train) jet-count
    fractions on a shared integer axis, plus a ratio panel.
    """
    target = _multiplicity(target_real)
    sample = _multiplicity(sample_real)

    length = max(target.max(), sample.max()) + 1     # Shared integer bins
    bins   = np.arange(length)
    frac_target = _fractions(target, length)
    frac_sample = _fractions(sample, length)

    ax.bar(bins, frac_target, width = 1.0, **TARGET)
    ax.bar(bins, frac_sample, width = 1.0, fill = False, edgecolor = SAMPLE_C, **SAMPLE)

    ratio_train = None
    if train_real is not None:          # Cut of dataset given?
        frac_train = _fractions(_multiplicity(train_real), length)
        ax.bar(bins, frac_train, width = 1.0, fill = False, edgecolor = TRAIN_C, **TRAIN)
        ratio_train = ratio(frac_sample, frac_train)

    plot_ratio(rax, bins, ratio(frac_sample, frac_target), ratio_train, step = True)
    finish_panel(ax, rax, "multiplicity", "Fraction of jets")
    rax.set_xticks(bins)                 # One tick per integer multiplicity


def plot_grid(
    target: Dataset,
    sample: dict[str, NDArray],
    channels: list[str] | None,
    n_events: int | None,
    panel: Callable,
    resolution: int,
) -> tuple[Figure, NDArray, list[str]]:
    """Shared flat/jet iteration behind `plot_histograms`/`plot_distributions`.
    Build the column grid (a leading multiplicity bar panel for jet data) and
    call `panel(ax, rax, name, truth, sampled, train, resolution)` per channel,
    where `resolution` is the histogram bin count or the KDE grid size. Return
    `(fig, axes, columns)`, where `columns` names what each column plots.
    """
    jet = "real" in sample
    truth, sampled, train = channel_data(target, sample, channels, n_events)

                                        # Leading multiplicity column for jets
    columns = ([ "multiplicity" ] if jet else []) + list(truth)
    offset  = len(columns) - len(truth)

    fig, axes = make_grid(len(columns))

    if jet:                             # First panel: multiplicity distribution
        train_real = None if n_events is None else target["real"][:n_events]
        multiplicity_panel(axes[0, 0], axes[1, 0], target["real"], sample["real"], train_real)

    for column, channel in enumerate(truth, start = offset):
        panel(axes[0, column], axes[1, column], channel,
              truth[channel], sampled[channel],
              None if train is None else train[channel], resolution)

    return fig, axes, columns


def finalise(fig: Figure, axes: NDArray, columns: list[str]) -> None:
    """Do final common actions on figure and axes: log-scale appropriate
    columns, then collect one figure-wide legend.
    """
    for column, name in enumerate(columns):
        if name in LOG_COLUMNS:
            axes[0, column].set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)
