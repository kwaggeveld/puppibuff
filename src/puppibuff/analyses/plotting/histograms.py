from __future__ import annotations

from ...datasets import Dataset
from .common import (SAMPLE, SAMPLE_C, TARGET, TRAIN, TRAIN_C,
                     finalise, finish_panel, plot_grid, plot_ratio, ratio)

import numpy as np

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#-----------------------------------------------------------------------------

def _hist(ax: Axes, values: NDArray, edges: NDArray, **kwargs) -> NDArray:
    """Draw one density histogram and return its bin counts. Mostly needed for 
    type checking.
    """
    counts, _, _ = ax.hist(values, bins = edges, density = True, **kwargs)  # type: ignore[arg-type]

    return np.asarray(counts)


def _panel(ax: Axes, rax: Axes, name: str, target: NDArray, sample: NDArray,
           training_cut: NDArray | None, bins: int) -> None:
    """One kinematic channel: target/sample/(optional train) density histograms
    on shared bin edges, plus a ratio panel.
    """
                                        # Shared bin edges so the two
                                        # histograms are directly comparable
    edges   = np.histogram_bin_edges(np.concatenate([target, sample]), bins = bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_target = _hist(ax, target, edges, histtype = "stepfilled", **TARGET)
    hist_sample = _hist(ax, sample, edges, histtype = "step", color = SAMPLE_C, **SAMPLE)

    ratio_train = None
    if training_cut is not None:        # Cut of dataset given?
        hist_train  = _hist(ax, training_cut, edges, histtype = "step",
                            color = TRAIN_C, **TRAIN)
        ratio_train = ratio(hist_sample, hist_train)

    plot_ratio(rax, centers, ratio(hist_sample, hist_target), ratio_train, step = True)
    finish_panel(ax, rax, name, "Probability density")


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
    fig, axes, columns = plot_grid(target, sample, channels, n_events, _panel, bins)
    finalise(fig, axes, columns)

    return fig
