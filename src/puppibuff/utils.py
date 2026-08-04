from __future__ import annotations

import numpy as np

from typing import Callable
from numpy.typing import NDArray

#-----------------------------------------------------------------------------

# Velocity field: f(t, x) = dx/dt, same shape as x.
Field = Callable[[float, NDArray], NDArray]

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


def midpoint_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        x_mid = x + .5 * h * f(t, x)
        x = x + h * f(t + .5 * h, x_mid)
        t += h

    return x