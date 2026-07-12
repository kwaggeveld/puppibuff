from typing import Callable

from numpy.typing import NDArray

# Velocity field: f(t, x) = dx/dt, same shape as x.
Field = Callable[[float, NDArray], NDArray]

def midpoint_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        x_mid = x + .5 * h * f(t, x)
        x = x + h * f(t + .5 * h, x_mid)
        t += h

    return x