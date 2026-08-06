from __future__ import annotations

from typing import Callable
from numpy.typing import NDArray

#-----------------------------------------------------------------------------

# Velocity field: f(t, x) = dx/dt, same shape as x.
Field = Callable[[float, NDArray], NDArray]

# Solve(f, x0, n_steps) -> x at t = 1.
Solver = Callable[[Field, NDArray, int], NDArray]


def euler_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    """Euler method of integration. Iterate y_{n + 1} = y_n + hf(t_n, y_n).
    One evaluation per interval.
    """
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        x = x + h * f(t, x)
        t += h

    return x


def ab2_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    """Two-step Adams-Bashforth: second order at Euler's one evaluation per
    interval. Since the second slope is the previous interval it can be reused.
    """
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    previous = None
    for _ in range(n_steps - 1):
        slope = f(t, x)
        x = x + h * (slope if previous is None else 1.5 * slope - .5 * previous)
        previous = slope
        t += h

    return x


def midpoint_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    """Midpoint method: two evaluations per interval."""
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        x_mid = x + .5 * h * f(t, x)
        x = x + h * f(t + .5 * h, x_mid)
        t += h

    return x


def heun_solve(f: Field, x0: NDArray, n_steps: int) -> NDArray:
    """Modified midpoint's method: two evaluations per interval, like 
    `midpoint_solve`, but evaluates BDTs a whole interval ahead which 
    is where they were trained.
    """
    h = 1. / (n_steps - 1)
    x = x0
    t = 0.
    for _ in range(n_steps - 1):
        k1 = f(t, x)
        k2 = f(t + h, x + h * k1)
        x = x + .5 * h * (k1 + k2)
        t += h

    return x
