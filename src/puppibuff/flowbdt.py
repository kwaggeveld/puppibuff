from __future__ import annotations

from .build_trainds import build_trainds, Paths
from .solvers import ab2_solve, Solver
from .utils import initial_noise, no_conditions, real_slots, t_to_step

from functools import partial

import numpy as np
from xgboost import XGBRegressor, XGBModel
import joblib
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlowBDT:
    def __init__(
        self,
        tree_config: dict | None = None,
        rng: np.random.Generator | None = None, # Draws `sample`'s noise
        max_multiplicity: int | None = None,    # Slots per jet, if required for conditioning
    ) -> None:
        self.tree_config = dict(tree_config or {})
        self.rng = rng
        self.max_multiplicity = max_multiplicity
                                        # Only if conditioned on multiplicity alone
        self.multiplicity_prior: NDArray | None = None


    def _fit_one(self, x: NDArray, y: NDArray, sample_weights: NDArray | None = None) -> XGBModel | None:
        if len(y) == 0:                 # Multiplicity slot not populated in
            return None                 # training dataset -> don't train/store

        return XGBRegressor(**self.tree_config).fit(x, y, sample_weight = sample_weights)


    def _predict_one(self, bdt: XGBModel | None, inputs: NDArray, channel: int) -> NDArray:
        if len(inputs) == 0:            # No sample has this column real
            return np.empty(0)

        if bdt is None:
            raise ValueError(f"No training sample had column { channel } real, "
                             f"so none can be sampled with it.")

        return bdt.predict(inputs)


    def _inputs(self, xt: NDArray, conditions: NDArray) -> NDArray:
        """`xt` with the `conditions` appended."""
        return np.column_stack([ xt, conditions ]).astype(np.float32)


    def _real_columns(self, conditions: NDArray) -> NDArray:
        """Each event's real columns as `(N, n_channels)`."""
        if self.max_multiplicity is None:
            return np.ones((len(conditions), self.n_channels), dtype = bool)

        slots = real_slots(conditions[:, 0], self.max_multiplicity)
        return np.tile(slots, self.n_channels // self.max_multiplicity)


    def _draw_conditions(self, n_samples: int, rng: np.random.Generator | None) -> NDArray:
        """Multiplicities from the training cut's histogram as `(n_samples, 1)`
        conditions if conditioned on the multiplicity alone, or none conditions 
        if conditioned on jet-level kinematics too, as these have to be 
        supplied externally.
        """
        if self.multiplicity_prior is None:
            return no_conditions(n_samples)

        rng = rng or self.rng or np.random.default_rng()
        prior = self.multiplicity_prior
        return rng.choice(len(prior), (n_samples, 1), p = prior)


    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None, # (N,), one weight per event
    ) -> None:
        # x: n_steps points along the paths, each (N, n_channels); y: (N, n_channels)
        self.n_steps      = x.n_steps
        self.n_channels   = y.shape[1]
        self.n_conditions = x.conditions.shape[1]

        if self.n_conditions == 1:      # Only multiplicity
            counts = np.bincount(x.conditions[:, 0].astype(int))
            self.multiplicity_prior = counts / counts.sum()

        real = self._real_columns(x.conditions)
                                        # Padding is paired with its own noise
        x = Paths(x.x0, np.where(real, x.x1, x.x0), x.ts, x.conditions)

        self._train_grid(x, y, real, sample_weights)


    def _train_grid(
        self,
        x: Paths,
        y: NDArray,
        real: NDArray,                  # (N, n_channels)
        sample_weights: NDArray | None,
    ) -> None:
        """Fit one BDT per (time step, channel) into `bdt_grid`."""
        if sample_weights is None:      # Uniform weights == unweighted fit
            sample_weights = np.ones(len(y))

        ensemble = []
        with tqdm(total = self.n_steps * self.n_channels,
                  desc  = "Training BDT grid") as progress_bar:
            for step in range(self.n_steps):
                xt = self._inputs(x[step], x.conditions)

                for channel in range(self.n_channels):
                                        # Only give samples where this slot is real
                    rows = real[:, channel]
                    ensemble.append(self._fit_one(xt[rows], y[rows, channel], sample_weights[rows]))
                    progress_bar.update()

        self.bdt_grid = (np.array(ensemble, dtype = object)
                           .reshape(self.n_steps, self.n_channels))


    def predict(self, t: float, xt: NDArray, conditions: NDArray | None = None) -> NDArray:
        """Velocity at time `t` at `xt`, both (N, n_channels)."""
        if conditions is None:
            conditions = no_conditions(len(xt))

        if conditions.shape[1] != self.n_conditions:
            raise ValueError(f"Expected { self.n_conditions } conditions per "
                             f"event, got { conditions.shape[1] }.")

        step     = t_to_step(t, self.n_steps)
        inputs   = self._inputs(xt, conditions)
        real     = self._real_columns(conditions)
        velocity = np.zeros_like(xt)    # Keep padding at zero velocity

        for channel, bdt in enumerate(self.bdt_grid[step]):
            rows = real[:, channel]
            velocity[rows, channel] = self._predict_one(bdt, inputs[rows], channel)

        return velocity


    def sample(
        self,
        n_samples: int | None = None,
        x0: NDArray | None = None,
        solver: Solver = ab2_solve,
        rng: np.random.Generator | None = None,
        *,
        conditions: NDArray | None = None,
    ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new sample. The `conditions`, drawn here
        unless given, are appended as trailing columns.
        """
        shape = None if n_samples is None else (n_samples, self.n_channels)
        x0    = initial_noise(shape, x0, rng or self.rng)

        if conditions is None:
            conditions = self._draw_conditions(len(x0), rng)

        conditioned_predict = partial(self.predict, conditions = conditions)
        raw = solver(conditioned_predict, x0, self.n_steps)

        return np.column_stack([ raw, conditions.astype(raw.dtype) ])

    def reflow(self, n_steps: int, n_pairs: int) -> FlowBDT:
        """Reflow this FlowBDT: train a FlowBDT with `n_steps` on `n_pairs` of 
        noise and generated samples from it.
        
        (2-rectified flow, arXiv:2209.03003).
        """
        x0 = initial_noise((n_pairs, self.n_channels), rng = self.rng)
        x1 = self.sample(x0 = x0)       # with trailing the conditions columns

        reflowed = FlowBDT(self.tree_config, self.rng, self.max_multiplicity)

        rectified_paths = build_trainds(x1[:, :self.n_channels], n_steps, x0,
                                        conditions = x1[:, self.n_channels:])
        reflowed.fit(*rectified_paths)

        return reflowed


# --- Export/import ---

    def save(self, path: str) -> None:
        """Persist the whole state except for `rng`, which is set by `load_model`,
        and export boosters to binary JSON.
        """
        state = vars(self) | { "rng": None }
        state["bdt_grid"] = [ None if bdt is None else bdt.get_booster().save_raw("ubj")
                              for bdt in self.bdt_grid.ravel() ]
        joblib.dump(state, path)

    def _load_one(self, raw_bdt: bytearray | None) -> XGBRegressor | None:
        if raw_bdt is None:             # Unused BDT, multiplicity not populated
            return None                 # in training cut

        bdt = XGBRegressor(**self.tree_config)
        bdt.load_model(raw_bdt)
        return bdt

    @classmethod
    def load(cls, path: str) -> FlowBDT:
        state    = joblib.load(path)
        raw_grid = state.pop("bdt_grid")    # Flat list of raw boosters

        obj = cls()
        obj.__dict__.update(state)
        obj.bdt_grid = np.array(
                           [ obj._load_one(raw_bdt) for raw_bdt in raw_grid ],
                           dtype = object
                       ).reshape(obj.n_steps, -1)

        return obj
