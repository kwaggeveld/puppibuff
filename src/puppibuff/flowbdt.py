from __future__ import annotations

from .build_trainds import Paths
from .solvers import midpoint_solve

import numpy as np
from xgboost import XGBRegressor, XGBModel
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm

from numpy.typing import NDArray



class FlowBDT():
    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def _fit_one(self, x: NDArray, y: NDArray, sample_weights: NDArray | None = None) -> XGBModel:
        return XGBRegressor(**self.config).fit(x, y, sample_weight = sample_weights)

    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None,
        n_threads: int = 1,
    ) -> None:
        # X: sequence of n_steps arrays, each (N, n_channels); y: (N, n_channels)
        self.n_steps    = x.n_steps
        self.n_channels = y.shape[1]

        ensemble = []
        with tqdm(total = self.n_steps * self.n_channels,
                  desc  = "Training BDT grid") as progress_bar:
            for step in range(self.n_steps):
                xt = x[step]            # Shared by every channel of this step

                jobs = (
                    delayed(self._fit_one)(xt, y[:, channel], sample_weights)
                    for channel in range(self.n_channels)
                )
                ensemble.extend(Parallel(n_jobs = n_threads)(jobs))

                progress_bar.update(self.n_channels)

                del xt

        self.bdt_grid = np.array(ensemble, dtype = object)
        self.bdt_grid = self.bdt_grid.reshape(self.n_steps, self.n_channels)

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
