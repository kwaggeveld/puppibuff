from __future__ import annotations

from .build_trainds import Paths
from .solvers import ab2_solve, Solver
from .utils import initial_noise, t_to_step

import numpy as np
from xgboost import XGBRegressor, XGBModel
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlowBDT():
    def __init__(
        self,
        config: dict | None = None,
        group_sizes: list[int] | None = None, # Widths of the column blocks
                                              # each BDT predicts
    ) -> None:
        self.config = dict(config or {}) # Need empty dict option for .from_json()
        self.group_sizes = list(group_sizes or []) # Empty => one BDT per channel


    @property
    def multi_output(self) -> bool:
        return self.config.get("multi_strategy") == "multi_output_tree"


    def _fit_one(self, x: NDArray, y: NDArray, sample_weights: NDArray | None = None) -> XGBModel:
        return XGBRegressor(**self.config).fit(x, y, sample_weight = sample_weights)


    def _prepare_targets(self, y: NDArray) -> list[NDArray]:
        """Default group_sizes to one group per channel, validate it against
        y's column count, then split y into per-group target blocks.
        """
        if not self.group_sizes:
            self.group_sizes = [1] * self.n_channels

        if sum(self.group_sizes) != self.n_channels:
            raise ValueError(
                f"Group sizes {self.group_sizes} sum to {sum(self.group_sizes)}, "
                f"but y has {self.n_channels} columns"
            )

        return np.split(y, np.cumsum(self.group_sizes)[:-1], axis = 1)


    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None, # (N,), one weight per event
        n_threads: int | None = None,          # None => one worker per group
    ) -> None:
        # X: sequence of n_steps arrays, each (N, n_channels); y: (N, n_channels)
        self.n_steps    = x.n_steps
        self.n_channels = y.shape[1]

        targets   = self._prepare_targets(y)
        n_threads = (len(targets) if n_threads is None
                     else min(n_threads, len(targets)))

        ensemble = []
        with tqdm(total = self.n_steps * len(targets),
                  desc  = "Training BDT grid") as progress_bar:
            for step in range(self.n_steps):
                xt = x[step]            # Shared by every group of this step

                jobs = (
                    delayed(self._fit_one)(xt, target, sample_weights)
                    for target in targets
                )

                for model in Parallel(n_jobs = n_threads, return_as = "generator")(jobs):
                    ensemble.append(model)
                    progress_bar.update()

        self.bdt_grid = (np.array(ensemble, dtype = object)
                            .reshape(self.n_steps, len(targets)))


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
        ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new event.
        """
        shape = None if n_samples is None else (n_samples, self.n_channels)
        x0    = initial_noise(shape, x0)

        return solver(self.predict, x0, self.n_steps)


# --- Export/import ---

    def to_disk(self, path: str) -> None:
        """Persist the whole state, and export boosters to binary JSON."""
        dump(self.__dict__ 
             | { "bdt_grid": [ bdt.get_booster().save_raw("ubj")
                               for bdt in self.bdt_grid.ravel() ]}, path)

    @classmethod
    def from_disk(cls, path: str) -> FlowBDT:
        obj = cls()
        obj.__dict__.update(load(path))

        grid = []
        for raw in obj.bdt_grid:        # Seeded from self.config: a raw booster
            bdt = XGBRegressor(**obj.config)  # carries no sklearn-side params
            bdt.load_model(raw)                                               # type: ignore
            grid.append(bdt)

        obj.bdt_grid = np.array(grid, dtype = object).reshape(obj.n_steps, -1)

        return obj
