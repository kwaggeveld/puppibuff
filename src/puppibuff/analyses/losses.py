from __future__ import annotations

from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def channel_mse(real: NDArray, gen: NDArray, bins: int = 100) -> float:
                                        # MSE between the two density
                                        # histograms, on shared bin edges
    edges = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)
    real_hist, _ = np.histogram(real, bins = edges, density = True)
    gen_hist,  _ = np.histogram(gen,  bins = edges, density = True)

    return float(np.mean((real_hist - gen_hist) ** 2))


def total_mse(
    data: Dataset,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    bins: int = 100,
) -> float:
    channels = channels or data.channels()

    losses = [                          # Ravel to flatten to 1D
        channel_mse(data[channel].ravel(), gen[channel].ravel(), bins = bins)
        for channel in channels
    ]
    return float(np.mean(losses))
