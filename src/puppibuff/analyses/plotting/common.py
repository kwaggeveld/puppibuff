from __future__ import annotations

from ...datasets import Dataset
from .style import DOC_WIDTH, LEGEND_LOC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from typing import Callable
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

Events = dict[str, NDArray]             # One distribution's per-channel arrays

#--- Styling defaults ---

TARGET_C, SAMPLE_C, TRAIN_C = "#9E9E9E", "#0C5DA5", "#FF9500"
# TARGET_C, SAMPLE_C, TRAIN_C = "tab:gray", "tab:blue", "tab:orange"

TARGET_EDGE = "#474747"
# TARGET_EDGE = "tab:brown"               # The fill is too pale to outline itself

TARGET: dict = dict(color = TARGET_C, edgecolor = TARGET_EDGE, alpha = 0.35,
                    linewidth = 0.5, zorder = 1, label = "Target")
SAMPLE: dict = dict(linewidth = 1.0, zorder = 3, label = "Output")
TRAIN:  dict = dict(linewidth = 0.8, zorder = 2, label = "Training")

TARGET_CONTOUR: dict = dict(colors = TARGET_EDGE, linestyles = "solid",  linewidths = 0.8, zorder = 1)
TRAIN_CONTOUR:  dict = dict(colors = TRAIN_C,     linestyles = "dotted", linewidths = 0.9, zorder = 3)
SAMPLE_CONTOUR: dict = dict(colors = SAMPLE_C,    linestyles = "dashed", linewidths = 1.1, zorder = 5)

                                        # `pt` spans orders of magnitude while                
LOG_CHANNELS = { "pt" }                 # eta/phi are O(1). Jets do as well.
LOG_COLUMNS  = LOG_CHANNELS | { "multiplicity" }

LABELS = {                              # Channel key to label
    "pt":           r"$p_\mathrm{T}$ [GeV]",
    "eta":          r"$\eta$",
    "phi":          r"$\phi$ [rad]",
    "multiplicity": r"Multiplicity",
}

LOG_LABELS = {                          # The `LOG_CHANNELS` rescale is a change of
                                        # variable, so the axis has to say so.
    "pt": r"$\log(1 + p_\mathrm{T}/\mathrm{GeV})$",
}


def label(name: str, log: bool = False) -> str:
    """Axis label for a channel, or the channel's own name if it has none. `log`
    returns for the `log1p`-rescaled form the contour plotters draw in.
    """
    if log:
        return LOG_LABELS.get(name, LABELS.get(name, name))

    return LABELS.get(name, name)

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


def _third_series(target: Dataset, overlay: dict[str, NDArray] | None,
                  n_events: int | None) -> Dataset | dict[str, NDArray] | None:
    """Source for the optional third series. Return the supplied `overlay`
    (a second decoded sample) if given, else the trained-on cut of `target`.
    """
    if overlay is not None:
        return overlay

    return None if n_events is None else target[:n_events]


def channel_data(
    target: Dataset,
    sample: dict[str, NDArray],
    channels: list[str] | None,
    n_events: int | None,
    overlay: dict[str, NDArray] | None = None,
) -> tuple[Events, Events, Events | None]:
    """Construct the `(target, sample, third)` arrays every plotter draws, with
    jet padding dropped. The third series is the trained-on cut of `target`, or
    a supplied `overlay`, or `None` when neither is given.
    """
    jet      = "real" in sample
    channels = channels or [ channel for channel in target.channels() if channel != "real" ]

    third = _third_series(target, overlay, n_events)

    return (_unpad(target, channels, target["real"] if jet else None),
            _unpad(sample, channels, sample["real"] if jet else None),
            None if third is None else _unpad(third, channels, third["real"] if jet else None))

#--- Panel scaffolding ---

PANEL_ASPECT = 1.25                     # Panel height (main + ratio) / width


def make_grid(n_cols: int, width: float) -> tuple[Figure, NDArray]:
    """Return a 2-row (main + ratio) * `n_cols` grid, `width` inches across."""
    fig, axes = plt.subplots(
        2, n_cols, figsize = (width, width / n_cols * PANEL_ASPECT),
        sharex = "col", gridspec_kw = { "height_ratios": [3, 1] },
    )
    fig.set_layout_engine("constrained", hspace = 0.0, wspace = 0.06)

    return fig, np.reshape(axes, (2, n_cols))


RATIO_BAND = 0.1                        # 1 +/- this fraction is shaded

RATIO_MAX = 3.0                         # ylim on the ratio axis

def _ratio_limits(*series: NDArray | None) -> tuple[float, float]:
    """Ratio range about 1, wide enough for the bulk of the points and capped."""
    finite = [ values[np.isfinite(values)] for values in series if values is not None ]
    values = np.concatenate(finite) if finite else np.empty(0)

    if values.size == 0:
        return 0.0, 2.0

    radius = float(np.clip(np.percentile(np.abs(values - 1.0), 99), 0.1, RATIO_MAX - 1.0))

    return max(0.0, 1.0 - radius), 1.0 + radius


def plot_ratio(rax: Axes, x: NDArray, sample_over_target: NDArray,
               training_cut: NDArray | None, step: bool) -> None:
    """Ratio panel: sample/target (and sample/train, if given), the unit line
    and a tolerance band. `step` picks stepped (histogram/bar) over smooth (KDE)
    rendering.
    """
    draw = rax.step if step else rax.plot

    step_kw: dict = { "where": "mid" } if step else {}

    rax.axhspan(1 - RATIO_BAND, 1 + RATIO_BAND, color = TARGET_C, alpha = 0.3, zorder = 0)
    rax.axhline(1.0, color = TARGET_EDGE, linestyle = "dashed", linewidth = 0.8, zorder = 1)

    if training_cut is not None:
        draw(x, training_cut, color = TRAIN_C, linewidth = TRAIN["linewidth"],
             zorder = 2, **step_kw)

    draw(x, sample_over_target, color = SAMPLE_C, linewidth = SAMPLE["linewidth"],
         zorder = 3, **step_kw)

    rax.set_ylim(*_ratio_limits(sample_over_target, training_cut))


XPAD = 0.1                             # Fraction of the span left blank at each end

def set_xlims(ax: Axes, low: float, high: float) -> None:
    """Pin the x axis to the data, with a little air at each end. Autoscaling
    spends the panel's width on whatever outlier is furthest out, while a bare
    `set_xlim` butts the first and last bin flat against the spine — so neither
    end is readable. This is the small margin between the two.
    """
    pad = (high - low) * XPAD

    ax.set_xlim(low - pad, high + pad)


def finish_panel(ax: Axes, rax: Axes, name: str, ylabel: str) -> None:
    """Shared axis labels for one main + ratio column."""
    ax.set_ylabel(ylabel)
    rax.set_ylabel("Ratio")
    rax.set_xlabel(label(name))


def _multiplicity(real: NDArray) -> NDArray:
    """Per-jet multiplicity as the row-sum of the real mask."""
    return real.sum(axis = 1).astype(int)


def _fractions(multiplicity: NDArray, length: int) -> NDArray:
    """Fraction of jets at each multiplicity 0 ... `length` - 1."""
    return np.bincount(multiplicity, minlength = length) / len(multiplicity)


def multiplicity_panel(ax: Axes, rax: Axes, target_real: NDArray,
                       sample_real: NDArray, train_real: NDArray | None) -> None:
    """Create bar chart of target/sample/(optional third series) jet-count
    fractions on a shared integer axis, plus a ratio panel. NB: the shared bin
    length spans target and sample only, so an `overlay` reaching a higher
    multiplicity than either would not fit the bars.
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
    finish_panel(ax, rax, "multiplicity", "Jet fraction")

                                        # Integer ticks
    rax.xaxis.set_major_locator(MaxNLocator(integer = True))


def plot_grid(
    target: Dataset,
    sample: dict[str, NDArray],
    channels: list[str] | None,
    n_events: int | None,
    panel: Callable,
    resolution: int,
    width: float,
    overlay: dict[str, NDArray] | None = None,
) -> tuple[Figure, NDArray, list[str]]:
    """Shared flat/jet iteration behind `plot_histograms`/`plot_distributions`.
    Build the column grid (a leading multiplicity bar panel for jet data) and
    call `panel(ax, rax, name, truth, sampled, third, resolution)` per channel,
    where `resolution` is the histogram bin count or the KDE grid size. Return
    `(fig, axes, columns)`, where `columns` names what each column plots.
    """
    jet = "real" in sample
    truth, sampled, train = channel_data(target, sample, channels, n_events, overlay)

                                        # Leading multiplicity column for jets
    columns = ([ "multiplicity" ] if jet else []) + list(truth)
    offset  = len(columns) - len(truth)

    fig, axes = make_grid(len(columns), width)

    if jet:                             # First panel: multiplicity distribution
        third = _third_series(target, overlay, n_events)
        multiplicity_panel(axes[0, 0], axes[1, 0], target["real"], sample["real"],
                           None if third is None else third["real"])

    for column, channel in enumerate(truth, start = offset):
        panel(axes[0, column], axes[1, column], channel,
              truth[channel], sampled[channel],
              None if train is None else train[channel], resolution)

    return fig, axes, columns


def share_labels(axes: NDArray) -> None:
    """Blank any ylabel that repeats the last one drawn in its row. Needed for
    jet plotting whose first column has a different ylabel than the rest.
    """
    for row in axes:
        drawn = ""

        for ax in row:
            if ax.get_ylabel() == drawn:
                ax.set_ylabel("")
            else:
                drawn = ax.get_ylabel()


def figure_legend(fig: Figure, handles: list, labels: list[str]) -> None:
    """One frameless figure-wide legend, in a strip below the axes."""
    fig.legend(handles, labels, loc = LEGEND_LOC, ncols = len(handles))


def finalise(fig: Figure, axes: NDArray, columns: list[str],
             labels: dict[str, str] | None = None) -> None:
    """Do final common actions on figure and axes: log-scale appropriate
    columns, then collect one figure-wide legend. `labels` renames series in
    that legend, keyed on their default label.
    """
    for column, name in enumerate(columns):
        if name in LOG_COLUMNS:
            axes[0, column].set_yscale("log")

    share_labels(axes)

    names: list[str]
    handles, names = axes[0, 0].get_legend_handles_labels()

    if labels:                          # Rename the series for one figure
        names = [ labels.get(name, name) for name in names ]

    figure_legend(fig, handles, names)
