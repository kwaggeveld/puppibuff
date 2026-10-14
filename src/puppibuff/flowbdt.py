from __future__ import annotations

from .build_trainds import Paths
from .solvers import ab2_solve, Solver
from .utils import initial_noise, t_to_step

import numpy as np
from xgboost import XGBRegressor, XGBModel
import joblib
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlowBDT():
    def __init__(
        self,
        tree_config: dict | None = None,
        rng: np.random.Generator | None = None, # Draws `sample`'s noise
    ) -> None:
        self.tree_config = dict(tree_config or {}) # Need empty dict option for .load()
        self.rng = rng


    def _fit_one(self, x: NDArray, y: NDArray, sample_weights: NDArray | None = None) -> XGBModel:
        return XGBRegressor(**self.tree_config).fit(x, y, sample_weight = sample_weights)


    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None, # (N,), one weight per event
    ) -> None:
        # X: sequence of n_steps arrays, each (N, n_channels); y: (N, n_channels)
        self.n_steps    = x.n_steps
        self.n_channels = y.shape[1]

        ensemble = []
        with tqdm(total = self.n_steps * self.n_channels,
                  desc  = "Training BDT grid") as progress_bar:
            for step in range(self.n_steps):
                xt = x[step]            # Shared by every channel of this step

                for channel in range(self.n_channels):
                    ensemble.append(self._fit_one(xt, y[:, channel], sample_weights))
                    progress_bar.update()

        self.bdt_grid = (np.array(ensemble, dtype = object)
                            .reshape(self.n_steps, self.n_channels))


    def predict(self, t: float, xt: NDArray) -> NDArray:
        # xt has shape (N, n_channels)
        step = t_to_step(t, self.n_steps)

                                        # (N, n_channels)
        return np.column_stack([bdt.predict(xt) for bdt in self.bdt_grid[step]])


    def sample(
            self,
            n_samples: int | None = None,
            x0: NDArray | None = None,
            solver: Solver = ab2_solve,
            rng: np.random.Generator | None = None,
        ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new event.
        """
        shape = None if n_samples is None else (n_samples, self.n_channels)
        x0    = initial_noise(shape, x0, rng if rng is not None else self.rng)

        return solver(self.predict, x0, self.n_steps)


# --- Export/import ---

    def save(self, path: str) -> None:
        """Persist the whole state except for `rng`, and export boosters to 
        binary JSON.
        """
        state = { key: value for key, value in self.__dict__.items() if key != "rng" }
        joblib.dump(state
                    | { "bdt_grid": [ bdt.get_booster().save_raw("ubj")
                                      for bdt in self.bdt_grid.ravel() ]}, path)

    @classmethod
    def load(cls, path: str) -> FlowBDT:
        obj = cls()
        obj.__dict__.update(joblib.load(path))

        grid = []
        for raw in obj.bdt_grid:
            bdt = XGBRegressor(**obj.tree_config)
            bdt.load_model(raw)                                               # type: ignore
            grid.append(bdt)

        obj.bdt_grid = np.array(grid, dtype = object).reshape(obj.n_steps, -1)

        return obj
