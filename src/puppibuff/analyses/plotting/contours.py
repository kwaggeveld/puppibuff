from __future__ import annotations

from ...datasets import Dataset
from .common import (LOG_CHANNELS, SAMPLE, SAMPLE_CONTOUR, TARGET, TARGET_CONTOUR,
                     TRAIN, TRAIN_CONTOUR, channel_data, kde)

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

QUANTILES = (.25, .5, .75, .95)         # Contours enclose these fractions of the
                                        # total probability mass

SPAN = (.1, 99.9)                       # Percentiles the evaluation grid spans.
                                        # Excluding extreme percentiles removes
                                        # outliers.

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


def _axis_grid(truth: NDArray, sampled: NDArray, points: int) -> NDArray:
    """Evaluation grid over one axis, spanning `SPAN` of the two clouds joined."""
    low, high = np.percentile(np.concatenate([truth, sampled]), SPAN)

    return np.linspace(low, high, points)


def _joint_density(marginals: tuple[NDArray, ...], grid: NDArray) -> NDArray:
    """Joint KDE of `marginals` evaluated over `grid`, a `(n_axes, *shape)`
    meshgrid, and reshaped back to that grid.
    """
    return kde(np.stack(marginals))(grid.reshape(len(marginals), -1)).reshape(grid.shape[1:])


def _contour(ax: Axes, grid: NDArray, density: NDArray, colors: str,
             linestyles: str, linewidths: float, zorder: int,
             fill: bool = False) -> None:
    """One series' mass contours. The keyword names are `contour`'s own, so a
    `*_CONTOUR` style dict splats straight in.
    """
    levels = _mass_levels(density)

    if fill:
        ax.contourf(*grid, density, levels = levels, colors = colors,
                    alpha = .25, zorder = zorder, extend = "max")

    ax.contour(*grid, density, levels = levels, colors = colors,
               linestyles = linestyles, linewidths = linewidths, zorder = zorder + 1)


def _rescale(names: tuple[str, ...], series: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
    """Map one point cloud into the space its KDE is estimated in: the logarithm
    for `LOG_CHANNELS`, the identity otherwise.
    """
    return tuple(np.log1p(values) if name in LOG_CHANNELS else values
                 for name, values in zip(names, series))


def _axis_label(name: str) -> str:
    """Name the plotted variable, which is the rescaled one for `LOG_CHANNELS`."""
    return f"log1p({ name })" if name in LOG_CHANNELS else name


def _pair_panel(
    ax: Axes,
    pair: tuple[str, str],
    target: tuple[NDArray, ...],
    sample: tuple[NDArray, ...],
    train: tuple[NDArray, ...] | None,
    points: int,
) -> None:
    """One channel pair: target/sample/(optional train) joint-KDE contours on a
    shared evaluation grid. The 2-D counterpart of `distributions`' panel.
    """
    target = _rescale(pair, target)
    sample = _rescale(pair, sample)
    train  = train if train is None else _rescale(pair, train)

                                        # Shared grid so the contour sets are
                                        # directly comparable
    axis_grids = [ _axis_grid(truth, sampled, points)
                   for truth, sampled in zip(target, sample) ]

    grid = np.stack(np.meshgrid(*axis_grids, indexing = "ij"))

    _contour(ax, grid, _joint_density(target, grid), fill = True, **TARGET_CONTOUR)

    if train is not None:               # Cut of dataset given?
        _contour(ax, grid, _joint_density(train, grid), **TRAIN_CONTOUR)

    _contour(ax, grid, _joint_density(sample, grid), **SAMPLE_CONTOUR)

    ax.set_xlabel(_axis_label(pair[0]))
    ax.set_ylabel(_axis_label(pair[1]))
    ax.grid(True, alpha = 0.3)


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
    truth, sampled, train = channel_data(target, sample, channels, n_events)
    pairs = list(combinations(truth, 2))

    fig, axes = plt.subplots(1, len(pairs), figsize = (5 * len(pairs), 5), squeeze = False)

    for ax, pair in zip(axes[0], pairs):
        _pair_panel(ax, pair,
                    tuple(truth[name]   for name in pair),
                    tuple(sampled[name] for name in pair),
                    None if train is None else tuple(train[name] for name in pair),
                    points)

                                        # A contour registers no legend handle,
                                        # so the labels ride on empty proxies.
    sets = [ (TARGET_CONTOUR, TARGET), (SAMPLE_CONTOUR, SAMPLE) ]

    if train is not None:               # Cut of dataset given?
        sets.append((TRAIN_CONTOUR, TRAIN))

    fig.legend(handles = [ Line2D([], [], color = style["colors"], label = kwargs["label"],
                                  linestyle = style["linestyles"])
                           for style, kwargs in sets ])

    return fig
