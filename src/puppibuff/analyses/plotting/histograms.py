from __future__ import annotations

from ...datasets import Dataset
from .common import (DOC_WIDTH, SAMPLE, SAMPLE_C, TARGET, TRAIN, TRAIN_C,
                     finalise, finish_panel, set_xlims, plot_grid, plot_ratio, ratio)

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
    """One kinematic channel: target/sample/(optional third series) density
    histograms on shared bin edges, plus a ratio panel.
    """
                                        # Shared bin edges so the histograms are
                                        # directly comparable
    series  = [ target, sample ] + ([] if training_cut is None else [ training_cut ])
    edges   = np.histogram_bin_edges(np.concatenate(series), bins = bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_target = _hist(ax, target, edges, histtype = "stepfilled", **TARGET)
    hist_sample = _hist(ax, sample, edges, histtype = "step", color = SAMPLE_C, **SAMPLE)

    ratio_train = None
    if training_cut is not None:        # Cut of dataset given?
        hist_train  = _hist(ax, training_cut, edges, histtype = "step",
                            color = TRAIN_C, **TRAIN)
        ratio_train = ratio(hist_sample, hist_train)

    plot_ratio(rax, centers, ratio(hist_sample, hist_target), ratio_train, step = True)
    finish_panel(ax, rax, name, "Density")
    set_xlims(ax, edges[0], edges[-1])


def plot_histograms(
    target: Dataset,                    # Truth channels (+ `real` for jets)
    sample: dict[str, NDArray],         # Decoded, generated channels
    channels: list[str] | None = None,
    n_events: int | None = None,        # Cut => overlay the trained-on subset
    bins: int = 75,
    width: float = DOC_WIDTH,           # Figure width in inches
    overlay: dict[str, NDArray] | None = None,  # A second sample, instead of the cut
    labels: dict[str, str] | None = None,       # Rename series, keyed on default label
) -> Figure:
    """Binned target/sample/(optional third series) distributions with ratio
    panels. Dispatches on `"real" in sample`: padded jet data gets a leading
    multiplicity bar panel and has its padding masked off; flat data gets one
    column per channel.

    The third series is the trained-on cut of `target` (`n_events`), or an
    `overlay` of decoded channels (eg. HLS sample).
    """
    fig, axes, columns = plot_grid(target, sample, channels, n_events, _panel, bins,
                                   width, overlay = overlay)
    finalise(fig, axes, columns, labels = labels)

    return fig
