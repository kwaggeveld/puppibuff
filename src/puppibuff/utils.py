from typing import Callable

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

# Velocity field: f(t, x) = dx/dt, same shape as x.
Field = Callable[[float, NDArray], NDArray]

def t_to_step(t: float, n_steps: int) -> int:
    """Snap `t` in [0, 1] to the nearest of `n_steps` integer time steps."""
    return int(np.floor(t * (n_steps - 1) + 0.5 + 1e-6))


def midpoint_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        x_mid = x + .5 * h * f(t, x)
        x = x + h * f(t + .5 * h, x_mid)
        t += h

    return x