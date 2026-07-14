import numpy as np

from numpy.typing import NDArray

EPSILON = 1e-3
                                        # x0.shape = x1.shape = (N, chs)
def build_paths(x0: NDArray, x1: NDArray, n_steps: int) -> NDArray:
    t = np.linspace(EPSILON, 1, num = n_steps,
                    dtype = np.float32).reshape(-1, 1, 1)          # (n_t, 1, 1)

    return t * x1 + (1. - t) * x0                            # (n_t, N, ch)
    # element [i, j, k] = t[i, 0, 0] * x1[j, k] through broadcasting

def build_trainds(x1: NDArray, n_steps: int) -> tuple[NDArray, NDArray]:
    x0 = np.random.normal(size = x1.shape).astype(np.float32)

    xt = build_paths(x0, x1, n_steps)   # Paths from noise to data (n_t, N, ch)

    vt = x1 - x0                        # Constant velocity along the paths, so
                                        # the target is shared by every step
    return xt, vt                       # (n_t, N, ch), (N, ch)