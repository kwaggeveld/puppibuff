from __future__ import annotations

from ...datasets import Dataset
from .common import (LOG_CHANNELS, SAMPLE, SAMPLE_C, TARGET, TRAIN, TRAIN_C,
                     finalise, finish_panel, kde, plot_grid, plot_ratio, ratio)

import numpy as np

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

def _density_ratio(sample: NDArray, ref: NDArray) -> NDArray:
    """`sample / ref`, blanked where the reference KDE drops below 0.1% of its
    peak: its exp-decaying tails reach ~0 far from any real support and would
    otherwise send the ratio to +/-inf (and overflow the divide).
    """
    return ratio(sample, np.where(ref < ref.max() * 1e-3, 0.0, ref))


def _panel(ax: Axes, rax: Axes, name: str, target: NDArray, sample: NDArray,
           train: NDArray | None, points: int) -> None:
    """One kinematic channel: target/sample/(optional train) Gaussian-KDE
    densities on a shared evaluation grid, plus a ratio panel. The smooth-curve
    counterpart of `histograms`' panel.
    """
                                        # Shared grid so the curves and their
                                        # ratio are directly comparable
    grid = np.linspace(min(target.min(), sample.min()),
                       max(target.max(), sample.max()), points)

    dens_target = kde(target)(grid)
    dens_sample = kde(sample)(grid)

    ax.fill_between(grid, dens_target, **TARGET)
    ax.plot(grid, dens_sample, color = SAMPLE_C, **SAMPLE)

    ratio_train = None
    if train is not None:               # Cut of dataset given?
        dens_train = kde(train)(grid)
        ax.plot(grid, dens_train, color = TRAIN_C, **TRAIN)
        ratio_train = _density_ratio(dens_sample, dens_train)

    plot_ratio(rax, grid, _density_ratio(dens_sample, dens_target), ratio_train, step = False)
    finish_panel(ax, rax, name, "Probability density")


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
    fig, axes, columns = plot_grid(target, sample, channels, n_events, _panel, points)
    finalise(fig, axes, columns)

                                        # A KDE never reaches 0, so its tails
                                        # break a log axis' autoscale. Pin each
                                        # log-scaled density panel from its peak
                                        # down to 8 orders below.
    for column, name in enumerate(columns):
        if name in LOG_CHANNELS:        # A KDE panel, not the multiplicity bars
            ax   = axes[0, column]
            peak = ax.dataLim.y1
            ax.set_ylim(peak * 1e-8, peak * 2)

    return fig
