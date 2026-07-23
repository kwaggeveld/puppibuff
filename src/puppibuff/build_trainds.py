import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

EPSILON = 1e-3

class Paths:
    """Array-like of shape (n_steps, N, n_channels) for lazy
    computation of each step xt on the path between x0 and x1"""

    def __init__(self, x0: NDArray, x1: NDArray, ts: NDArray) -> None:
        self.x0 = x0                    # (n_events, n_channels)
        self.x1 = x1                    # (n_events, n_channels)
        self.ts = ts                    # (n_steps)
        self.n_steps = len(ts)

    def __getitem__(self, step: int) -> NDArray:
        t = self.ts[step]
        return t * self.x1 + (1. - t) * self.x0

def build_trainds(x1: NDArray, n_steps: int) -> tuple[Paths, NDArray]:
    """Draw noise x0 and build Paths from `x0` to `x1` with `n_steps` time steps."""
    x0 = np.random.default_rng().standard_normal(x1.shape, dtype = np.float32)

    ts = np.linspace(EPSILON, 1, num = n_steps, dtype = np.float32)

    return Paths(x0, x1, ts), x1 - x0
