from __future__ import annotations

from .solvers import midpoint_solve

from xgboost import XGBRegressor, XGBModel
from joblib import delayed, dump, load
import numpy as np

from numpy.typing import NDArray

from tqdm_joblib import ParallelPbar


class FlowBDT():
    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def _fit_one(self, x: NDArray, y: NDArray) -> XGBModel:
        return XGBRegressor(**self.config).fit(x, y)

    def fit(self, xt: NDArray, yt: NDArray, n_threads: int = 1) -> None:
        # xt has shape `(n_steps, N, n_channels)`; yt is the velocity target,
        # constant along each path and so shared by every step: `(N, n_channels)`

        self.n_steps, _, self.n_channels = xt.shape

        jobs = (                        # One BDT per (step, channel)
            delayed(self._fit_one)(xt[step], yt[:, channel])
            for step    in range(self.n_steps)
            for channel in range(self.n_channels)
        )

        ensemble = ParallelPbar("Training BDT grid")(n_jobs = n_threads)(jobs)
        self.bdt_grid = np.array(ensemble, dtype = object)
        self.bdt_grid = self.bdt_grid.reshape(self.n_steps, self.n_channels)

        print("Training done")

    def predict(self, t: float, xt: NDArray) -> NDArray:
        # xt has shape (N, n_channels)
                                        # Convert t in [0, 1] to integer step
        step = int(np.floor(t * (self.n_steps - 1) + 0.5 + 1e-6)) 

        ret = np.empty_like(xt)         # Call approp. BDT for each channel
        for channel in range(self.n_channels):
            ret[:, channel] = self.bdt_grid[step, channel].predict(xt)
        return ret                      # (N, n_channels)

    def sample(self, n_samples: int) -> NDArray:
        x0 = np.random.normal(size = (n_samples, self.n_channels)).astype(np.float32)
        return midpoint_solve(self.predict, x0, self.n_steps)

# --- Export/import ---

    def to_disk(self, path: str) -> None:
        dump(self.__dict__, path)

    @classmethod
    def from_disk(cls, path: str) -> FlowBDT:
        obj = cls()
        obj.__dict__.update(load(path))

        return obj
