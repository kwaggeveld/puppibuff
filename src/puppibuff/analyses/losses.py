from __future__ import annotations

from ..datasets import Dataset

import numpy as np
from scipy.stats import wasserstein_distance

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def channel_mse(real: NDArray, gen: NDArray, bins: int = 75) -> float:
    """Compute MSE error between `real` and `gen` histograms."""
    edges = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)
    real_hist, _ = np.histogram(real, bins = edges, density = True)
    gen_hist,  _ = np.histogram(gen,  bins = edges, density = True)

    return float(np.mean((real_hist - gen_hist) ** 2))


def total_mse(
    data: Dataset,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    bins: int = 75,
) -> float:
    """Compute mean MSE loss across all channels."""
    channels = channels or data.channels()

    losses = [
        channel_mse(data[channel], gen[channel], bins = bins)
        for channel in channels
    ]
    return float(np.mean(losses))


def channel_wasserstein(real: NDArray, gen: NDArray) -> float:
    """1-D Wasserstein distance between the `real` and `gen` samples of a
    single channel. Per-channel, so does not measure MD correlation.
    """
    return float(wasserstein_distance(real, gen))

#--- Joint metric (correlation-aware) ---

def joint_mse(
    data: Dataset,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    bins: int = 20,
) -> float:
    """MSE between the joint real and generated densities on a shared
    `histogramdd` grid: the multi-dimensional counterpart of `total_mse`.
    Its absolute value depends on `bins`, but at a fixed `bins` it is 
    deterministic and is appropriate to rank models with.
    """
    channels = channels or data.channels()
                                        # (N, d) point clouds
    real = np.stack([ data[channel] for channel in channels ], axis = 1)
    fake = np.stack([ gen[channel]  for channel in channels ], axis = 1)

                                        # Shared per-axis edges over the combined
                                        # range, so both densities share one grid
    edges = [np.histogram_bin_edges(np.concatenate([real[:, axis], fake[:, axis]]), bins = bins)
             for axis in range(len(channels))]

    real_hist, _ = np.histogramdd(real, bins = edges, density = True)
    fake_hist, _ = np.histogramdd(fake, bins = edges, density = True)

    return float(np.mean((real_hist - fake_hist) ** 2))
