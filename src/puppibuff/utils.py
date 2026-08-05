from __future__ import annotations

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

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
