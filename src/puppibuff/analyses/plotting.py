from __future__ import annotations

from ..utils import Source, flatten

from functools import partial
from itertools import combinations
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from scipy.stats import gaussian_kde

from typing import Callable
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------
            # Supresses error:
            #   'created' timestamp seems very low; regarding as unix timestamp
            #   'modified' timestamp seems very low; regarding as unix timestamp
logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)

#--- Styling defaults ---               # rcParams are in `puppibuff/style.mplstyle`

DOC_WIDTH = 0.9 * 6.3                   # Typical article `\the\textwidth` / 72.28

STYLES: list[dict] = [                  # The target, then each sample
    dict(color = "#9E9E9E", edgecolor = "#474747", alpha = .35, linewidth = .5, zorder = 1),
    dict(color = "#0C5DA5", linewidth = 1.0, zorder = 3),
    dict(color = "#FF9500", linewidth = .8, zorder = 2),
]
TARGET = STYLES[0]

LABELS = {                              # Channel key to label
    "pt":           r"$p_\mathrm{T}$ / GeV",
    "eta":          r"$\eta$",
    "phi":          r"$\phi$ / rad",
    "multiplicity": "Multiplicity",
}
                                        # Contours estimate these channels'
LOG_LABELS = {                          # KDEs in log1p space
    "pt": r"$\log(1 + p_\mathrm{T} / \mathrm{GeV})$",
}
LOG_Y = { "pt", "multiplicity" }        # Marginals spanning orders of magnitude

PANEL_ASPECT   = 0.95                   # Axes height (main + ratio) / width
XPAD           = 0.1                    # Fraction of the span left blank at each end
RATIO_BAND     = 0.1                    # 1 +/- this fraction is shaded
RATIO_MAX      = 3.0                    # ylim on the ratio axis
KDE_MAX_POINTS = 50_000                 # A KDE eval is O(n_data * len(grid))
QUANTILES      = (.25, .5, .75, .95)    # Fractions of the mass contours enclose
SPAN           = (.1, 99.9)             # Percentiles a contour grid spans

#--- Data ---

def _series(target: Source, samples: dict[str, Source] | None,
            channels: list[str] | None) -> tuple[list[str], list[dict[str, NDArray]]]:
    """Return legend labels and flattened channels of the target and samples."""
    samples = samples or {}

    if len(samples) >= len(STYLES):
        raise ValueError(f"Can plot at most { len(STYLES) - 1 } samples, "
                         f"got { len(samples) }.")

    channels = channels or [ channel for channel in target if channel != "real" ]

    return ([ "Target", *samples ],
            [ flatten(source, channels) for source in (target, *samples.values()) ])


def _kde(data: NDArray) -> gaussian_kde:
    """Return Gaussian KDE estimator for `data`, subsampling large inputs.
    `data` is 1D for a marginal, or `(n_channels, n_events)` for a joint
    density. NB: transposed compared to normal.
    """
    n_points = data.shape[-1]
    if n_points > KDE_MAX_POINTS:
        data = data[..., np.random.default_rng(0).choice(n_points, KDE_MAX_POINTS, replace = False)]

    return gaussian_kde(data)

#--- Ratio row ---

def _ratio(num: NDArray, den: NDArray, floor: float = 0.) -> NDArray:
    """`num / den`, NaN where `den` is zero or under `floor` of its peak."""
    den = np.where(den < den.max() * floor, 0., den)

    return np.divide(num, den, out = np.full_like(num, np.nan), where = den > 0)


def _ratio_panel(rax: Axes, grid: NDArray, curves: list[NDArray], step: bool,
                 floor: float = 0.) -> None:
    """Ratio row; the first sample against the target and other curves.
    Draw histograms if `step`, else smooth curves.
    """
    target, first, *others = curves
    draw = partial(rax.step, where = "mid") if step else rax.plot

    rax.axhspan(1 - RATIO_BAND, 1 + RATIO_BAND, color = TARGET["color"], alpha = .3, zorder = 0)
    rax.axhline(1., color = TARGET["edgecolor"], linestyle = "dashed", linewidth = .8, zorder = 1)

    ratios = [ _ratio(first, ref, floor) for ref in (target, *others) ]

    for ratio, style in zip(ratios, STYLES[1:]):
        draw(grid, ratio, **style)

    values = np.concatenate([ ratio[np.isfinite(ratio)] for ratio in ratios ])
    radius = (np.clip(np.percentile(np.abs(values - 1.), 99), 0.1, RATIO_MAX - 1.0)
              if values.size else 1.0)

    rax.set_ylim(max(0.0, 1.0 - radius), 1.0 + radius)

#--- Marginals ---

def _hist_panel(ax: Axes, rax: Axes | None, name: str, values: list[NDArray],
                labels: list[str], bins: int = 75) -> None:
    """Draw density histograms on shared edges with unit-wide bins for 
    `multiplicity`."""
    if name == "multiplicity":
        edges = np.arange(max(series.max() for series in values) + 2) - .5
    else:
        edges = np.histogram_bin_edges(np.concatenate(values), bins = bins)

    counts = [ np.asarray(ax.hist(series, edges, density = True, label = label,     # type: ignore[arg-type]
                                  histtype = "stepfilled" if index == 0 else "step",
                                  **STYLES[index])[0])
               for index, (series, label) in enumerate(zip(values, labels)) ]

    if rax is not None:
        _ratio_panel(rax, (edges[:-1] + edges[1:]) / 2, counts, step = True)


def _kde_panel(ax: Axes, rax: Axes | None, name: str, values: list[NDArray],
               labels: list[str], points: int = 200) -> None:
    """Draw Gaussian-KDE densities on a shared grid with `multiplicity` a histogram."""
    if name == "multiplicity":
        return _hist_panel(ax, rax, name, values, labels)

    grid      = np.linspace(min(series.min() for series in values),
                            max(series.max() for series in values), points)
    densities = [ _kde(series)(grid) for series in values ]

    ax.fill_between(grid, densities[0], label = labels[0], **TARGET)
    for index in range(1, len(densities)):
        ax.plot(grid, densities[index], label = labels[index], **STYLES[index])

    if rax is not None:                 # KDE tails are long and ~~0, cut below 1e-3
        _ratio_panel(rax, grid, densities, step = False, floor = 1e-3)

    if name in LOG_Y:                   # Manually set lower bound for log plots
        peak = ax.dataLim.y1            # as well, otherwise log(~0) goes -infty
        ax.set_ylim(peak * 1e-8, peak * 2)


def _marginals(target: Source, samples: dict[str, Source] | None,
               channels: list[str] | None, width: float, panel: Callable) -> Figure:
    """One panel per flattened channel, with a ratio row beneath when there are
    samples. Four panels are displayed in a 2 x 2 grid.
    """
    labels, series = _series(target, samples, channels)
    names   = list(series[0])
    heights = [ 3, 1 ] if len(series) > 1 else [ 3 ]
    n_cols  = 2 if len(names) == 4 else len(names)
    n_rows  = len(names) // n_cols

    fig, axes = plt.subplots(
        len(heights) * n_rows, n_cols, squeeze = False,
        figsize = (width, width / n_cols * PANEL_ASPECT * sum(heights) / 4 * n_rows),
        gridspec_kw = { "height_ratios": heights * n_rows },
    )
    fig.set_layout_engine("constrained", hspace = 0., wspace = .06)

    for index, name in enumerate(names):
        row, column = divmod(index, n_cols)
        ax  = axes[len(heights) * row, column]
        rax = axes[len(heights) * row + 1, column] if len(heights) > 1 else None

        if rax is not None:             # Ratio row hangs off its main panel
            rax.sharex(ax)
            ax.tick_params(axis = "x", which = "both", labelbottom = False)

        panel(ax, rax, name, [ columns[name] for columns in series ], labels)

        low, high = ax.dataLim.intervalx
        ax.set_xlim(low - (high - low) * XPAD, high + (high - low) * XPAD)

        if name in LOG_Y:
            ax.set_yscale("log")
        if name == "multiplicity":      # Integer ticks
            ax.xaxis.set_major_locator(MaxNLocator(integer = True))

        (ax if rax is None else rax).set_xlabel(LABELS.get(name, name))

        if column == 0:
            ax.set_ylabel("Density")
            if rax is not None:
                rax.set_ylabel("Ratio")

    fig.legend(*axes[0, 0].get_legend_handles_labels(),
               loc = "outside upper right", ncols = len(series))

                                        # Recalculate required figure size
                                        # for equal aspect ratio on main panels
    for _ in range(2):                  # no matter what's plotted
        fig.draw_without_rendering()    # (twice as tick labels change with height)
        fig_width, fig_height = fig.get_size_inches()
        axes_width  = axes[0, 0].get_position().width * fig_width
        axes_height = sum(ax.get_position().height for ax in axes[:, 0]) * fig_height

        fig.set_size_inches(fig_width, fig_height - axes_height
                            + n_rows * PANEL_ASPECT * axes_width * sum(heights) / 4)

    return fig

#--- Contours ---

def _mass_levels(density: NDArray) -> NDArray:
    """Density levels enclosing `QUANTILES` of the total probability mass,
    ascending as `contour` requires. Sorting descending turns the cumulative sum
    into "mass at or above this level", so a quantile's level is where that first
    reaches it. `np.unique` both sorts and drops the duplicates two quantiles
    produce on a flat density, which `contour` rejects.
    """
    ranked   = np.sort(density, axis = None)[::-1]
    enclosed = np.cumsum(ranked) / ranked.sum()

    return np.unique(ranked[np.clip(np.searchsorted(enclosed, QUANTILES), 0, ranked.size - 1)])

#--- Public ---

def plot_histograms(
    target: Source,                     # Truth channels (+ `real` for jets)
    samples: dict[str, Source] | None = None,   # Legend label to channels
    *,
    channels: list[str] | None = None,
    bins: int = 75,
    width: float = DOC_WIDTH,           # Figure width in inches
) -> Figure:
    """Histogram marginals of `target` and each labelled sample, with a ratio
    row of the first sample against every other series. Constituent-level jet
    samples are displayed with a multiplicity panel.
    """
    return _marginals(target, samples, channels, width, partial(_hist_panel, bins = bins))


def plot_distributions(
    target: Source,                     # Truth channels (+ `real` for jets)
    samples: dict[str, Source] | None = None,   # Legend label to channels
    *,
    channels: list[str] | None = None,
    points: int = 200,                  # KDE evaluation-grid resolution
    width: float = DOC_WIDTH,           # Figure width in inches
) -> Figure:
    """`plot_histograms` with Gaussian-KDE curves instead of histograms"""
    return _marginals(target, samples, channels, width, partial(_kde_panel, points = points))


def plot_contours(
    target: Source,                     # Truth channels (+ `real` for jets)
    samples: dict[str, Source] | None = None,   # Legend label to channels
    *,
    channels: list[str] | None = None,
    points: int = 100,                  # KDE evaluation-grid resolution per axis
    width: float = DOC_WIDTH,           # Figure width in inches
) -> Figure:
    """Pairwise joint-KDE contours, each enclosing `QUANTILES` of the mass, on
    square panels. `LOG_LABELS` channels are estimated in log1p space.
    """
    labels, series = _series(target, samples, channels)
                                        # Per-jet, so not aligned with constituents
    pairs = list(combinations([ name for name in series[0] if name != "multiplicity" ], 2))

    fig, axes = plt.subplots(1, len(pairs), figsize = (width, width / len(pairs)),
                             squeeze = False, layout = "constrained")

    for ax, pair in zip(axes[0], pairs):
        clouds = [ np.stack([ np.log1p(columns[name]) if name in LOG_LABELS else columns[name]
                              for name in pair ])
                   for columns in series ]
                                        # Grid spans the target + first sample,
        spanned   = np.concatenate(clouds[:2], axis = 1)
        low, high = np.percentile(spanned, SPAN, axis = 1)
        pad       = (high - low) * XPAD # padded so the contours close
        grid      = np.stack(np.meshgrid(*np.linspace(low - pad, high + pad, points, axis = 1),
                                         indexing = "ij"))

        for index, cloud in enumerate(clouds):
            density = _kde(cloud)(grid.reshape(2, -1)).reshape(grid.shape[1:])
            levels  = _mass_levels(density)
            style   = STYLES[index]

            if index == 0:              # Target shaded darker towards its core
                shades = [ to_rgba(style["edgecolor"], alpha) for alpha in np.linspace(.15, .6, len(levels)) ]
                ax.contourf(*grid, density, levels = levels, colors = shades,
                            extend = "max", zorder = style["zorder"])
            else:
                ax.contour(*grid, density, levels = levels, colors = style["color"],
                           linewidths = style["linewidth"], zorder = style["zorder"])

                                        # Fit the panel to the drawn contours
        bounds = Bbox.union([ drawn.get_datalim(ax.transData) for drawn in ax.collections ])
        bounds = bounds.padded(bounds.width * XPAD, bounds.height * XPAD)
        ax.set(xlim = bounds.intervalx, ylim = bounds.intervaly)

        ax.set_xlabel(LOG_LABELS.get(pair[0]) or LABELS.get(pair[0], pair[0]))
        ax.set_ylabel(LOG_LABELS.get(pair[1]) or LABELS.get(pair[1], pair[1]))

                                        # A contour has no legend handle
    proxies = [ Patch(color = TARGET["edgecolor"], alpha = .4),
                *[ Line2D([], [], **style) for style in STYLES[1:len(series)] ] ]

    fig.legend(proxies, labels, loc = "outside upper right", ncols = len(series))

    return fig
