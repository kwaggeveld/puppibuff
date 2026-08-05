from __future__ import annotations

from .datasets import Dataset
from .codecs import Codec
from .build_trainds import build_trainds, Paths
from .configs import Config

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def setup_from_config(config: Config) -> tuple[Dataset, Codec, FlowBDT, Paths, NDArray]:
    """Load the dataset, fit + apply the codec, build the flow-matching
    training set, and construct the (untrained) model using the config."""
    from .flowbdt import FlowBDT        # Deferred: `flowbdt` imports this module

    data = config.dataset_cls()

    codec = config.codec_cls(config.s1phi)
    codec.fit(data)

    x1 = codec.encode(data[:config.n_events])
    x, y = build_trainds(x1, config.n_steps)

    sizes = codec.group_sizes() if config.multi_output else None
    model = FlowBDT(config.tree_config, sizes)

    return data, codec, model, x, y


def t_to_step(t: float, n_steps: int) -> int:
    """Snap `t` in [0, 1] to the nearest of `n_steps` integer time steps."""
    return int(np.floor(t * (n_steps - 1) + 0.5 + 1e-6))


def initial_noise(
        n_samples: int | None,
        n_channels: int,
        x0: NDArray | None = None,
    ) -> NDArray:
    """Return noise a sampler starts from, drawn here unless `x0` provides it."""
    if x0 is not None:
        return x0

    if n_samples is None:
        raise ValueError("Provide either n_samples or initial noise x0.")

    return np.random.normal(size = (n_samples, n_channels)).astype(np.float32)
