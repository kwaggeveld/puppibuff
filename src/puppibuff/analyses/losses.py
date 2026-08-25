from __future__ import annotations

from ..datasets import Dataset
from ..configs.config import DEFAULT_TREE_CONFIG

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

Source = Dataset | dict[str, NDArray]   # Anything indexable by channel name


def _channels(data: Source, channels: list[str] | None) -> list[str]:
    """Explicit channel list, else every channel `data` carries."""
    if channels is not None: 
        return channels
    return data.channels() if isinstance(data, Dataset) else list(data)


def channel_mse(real: NDArray, gen: NDArray, bins: int = 75) -> float:
    """Compute MSE error between `real` and `gen` histograms."""
    edges = np.histogram_bin_edges(np.concatenate([real, gen]), bins = bins)
    real_hist, _ = np.histogram(real, bins = edges, density = True)
    gen_hist,  _ = np.histogram(gen,  bins = edges, density = True)

    return float(np.mean((real_hist - gen_hist) ** 2))


def total_mse(
    data: Source,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    bins: int = 75,
) -> float:
    """Compute mean MSE loss across all channels."""
    channels = _channels(data, channels)

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
    data: Source,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    bins: int = 20,
) -> float:
    """MSE between the joint real and generated densities on a shared
    `histogramdd` grid: the multi-dimensional counterpart of `total_mse`.
    Its absolute value depends on `bins`, but at a fixed `bins` it is 
    deterministic and is appropriate to rank models with.
    """
    channels = _channels(data, channels)
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


SLICED_SEED = 0


def _subsample(cloud: NDArray, n: int, rng: np.random.Generator) -> NDArray:
    """Draw `n` rows without replacement if there are more than `n` rows."""
    if len(cloud) <= n: return cloud
    return cloud[rng.choice(len(cloud), n, replace = False)]


def sliced_wasserstein(
    data: Source,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    n_projections: int = 128,
    max_events: int | None = 500_000,
) -> float:
    """Mean 1-D Wasserstein distance over `n_projections` random projection
    directions.

    Channels are standardised to prevent `pt`'s ~1000 GeV range from dominating
    the metric. Both clouds are subsampled to `max_events` (`None` for all).
    """
    channels = _channels(data, channels)
                                        # (N, d) point clouds, as `joint_mse` builds
    real = np.stack([ data[channel] for channel in channels ], axis = 1)
    fake = np.stack([ gen[channel]  for channel in channels ], axis = 1)

    rng = np.random.default_rng(SLICED_SEED)

                                        # Uniform on the unit sphere
    directions  = rng.normal(size = (n_projections, len(channels)))
    directions /= np.linalg.norm(directions, axis = 1, keepdims = True)

    if max_events is not None:
        real = _subsample(real, max_events, rng)
        fake = _subsample(fake, max_events, rng)

    mean, std = real.mean(axis = 0), real.std(axis = 0)
    real, fake = (real - mean) / std, (fake - mean) / std

    return float(np.mean([
        wasserstein_distance(real @ direction, fake @ direction)
        for direction in directions
    ]))

#--- Classifier two-sample test ---

def _c2st_auc(
    x0: NDArray,
    x1: NDArray,
    test_size: float,
    tree_config: dict
) -> float:
    """Train a classifier to distinguish between two point clouds."""
    features = np.concatenate([x0, x1])
    labels   = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))])

                                        # Stratisfied so that labels keep same 
                                        # relative frequency
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels,
        test_size = test_size, stratify = labels,
    )

    config = tree_config | { "objective": "binary:logistic", "eval_metric": "auc" }
    model  = XGBClassifier(**config).fit(train_x, train_y)

    return float(roc_auc_score(test_y, model.predict_proba(test_x)[:, 1]))


def _quantise(values: NDArray, step: float) -> NDArray:
    """Round into a grid of size `step`."""
    return np.round(values / step) * step


def classifier_two_sample_test(
    data: Source,
    gen: dict[str, NDArray],
    channels: list[str] | None = None,
    test_size: float = .2,
    tree_config: dict | None = None,
    quantisation: dict[str, float] | None = None,
) -> float:
    """Joint discrepancy as the AUC score of a classifier trained to tell the
    real and generated events apart. .5 is indistinguishable, 1 perfectly 
    separable. Correlation-aware like `joint_mse`, but grid-free, so it does not
    thin out as channels are added.

    `quantisation` specifies a grid that `data` and `gen` will be snapped to. Without,
    the classifier learns to distinguish based on quantisation. On `FlatPuppiJet` 
    it costs 0.43 of AUC: `pt` reads 0.92 raw vs 0.56 snapped.
    """
    channels = _channels(data, channels)
                                        # (N, d) point clouds, as `joint_mse` builds
    real = np.stack([ data[channel] for channel in channels ], axis = 1)
    fake = np.stack([ gen[channel]  for channel in channels ], axis = 1)

    for axis, channel in enumerate(channels):
        quantum = (quantisation or {}).get(channel)
        if quantum is None: continue

        real[:, axis] = _quantise(real[:, axis], quantum)
        fake[:, axis] = _quantise(fake[:, axis], quantum)

    n_samples = min(len(real), len(fake))
    rng = np.random.default_rng(0)

    if len(real) > n_samples:
        real = real[rng.choice(len(real), n_samples, replace = False)]
    if len(fake) > n_samples:
        fake = fake[rng.choice(len(fake), n_samples, replace = False)]

    tree_config = dict(tree_config if tree_config is not None else DEFAULT_TREE_CONFIG)

    return _c2st_auc(real, fake, test_size, tree_config)
