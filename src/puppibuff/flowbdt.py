from __future__ import annotations

from .build_trainds import build_trainds, Paths
from .solvers import ab2_solve, Solver
from .utils import initial_noise, t_to_step

from functools import partial

import numpy as np
from xgboost import XGBRegressor, XGBModel
import joblib
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def slot_mask(multiplicities: NDArray, n_columns: int, max_multiplicity: int) -> NDArray:
    """(N, n_columns) mask of real constituents."""
                                        # Columns are channel-major,
                                        # so first `multiplicity` of every 
                                        # `max_multiplicity` slots are real
    return np.arange(n_columns) % max_multiplicity < multiplicities[:, None]


class FlowBDT:
    def __init__(
        self,
        tree_config: dict | None = None,
        rng: np.random.Generator | None = None, # Draws `sample`'s noise
    ) -> None:
        self.tree_config = dict(tree_config or {}) # Need empty dict option for .load()
        self.rng = rng
                                        # Set by `fit` if conditioned
        self.max_multiplicity: int | None = None
                                        # Only if conditioned on multiplicity alone
        self.multiplicity_prior: NDArray | None = None


    def _fit_one(self, x: NDArray, y: NDArray, sample_weights: NDArray | None = None) -> XGBModel | None:
        if len(y) == 0:                 # Multiplicity slot not populated in
            return None                 # training dataset -> don't train/store

        return XGBRegressor(**self.tree_config).fit(x, y, sample_weight = sample_weights)


    def _predict_one(self, bdt: XGBModel | None, inputs: NDArray, slot: int) -> NDArray:
        if len(inputs) == 0:            # No sampled jet populates this slot
            return np.empty(0)

        if bdt is None:
            raise ValueError(f"No training jet had more than { slot } constituents, "
                             f"so none can be sampled with more.")

        return bdt.predict(inputs)


    def _inputs(self, xt: NDArray, conditions: NDArray | None) -> NDArray:
        """`xt` with the `conditions` appended if given."""
        if conditions is None:
            return xt

        return np.column_stack([ xt, conditions ]).astype(np.float32)


    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None, # (N,), one weight per event
    ) -> None:
        # x: n_steps points along the paths, each (N, n_channels); y: (N, n_channels)
        self.n_steps    = x.n_steps
        self.n_channels = y.shape[1]

        conditions = x.conditions
        mask: NDArray | None = None
        if conditions is not None:      # Constituents: rebuild Paths
            if x.max_multiplicity is None:
                raise ValueError("Conditioned Paths need a max_multiplicity.")

            self.max_multiplicity = x.max_multiplicity
            multiplicities = conditions[:, 0].astype(int)
            
            mask = slot_mask(multiplicities, self.n_channels, self.max_multiplicity)

            if conditions.shape[1] == 1:# Only multiplicity
                counts = np.bincount(multiplicities, minlength = self.max_multiplicity + 1)
                self.multiplicity_prior = counts / counts.sum()
                                        # Padding is paired with its own noise
            x = Paths(x.x0, np.where(mask, x.x1, x.x0), x.ts)

        self._train_grid(x, y, conditions, mask, sample_weights)


    def _train_grid(
        self,
        x: Paths,
        y: NDArray,
        conditions: NDArray | None,     # (N, n_conditions), if conditioned
        mask: NDArray | None,           # (N, n_channels) real slots, if conditioned
        sample_weights: NDArray | None,
    ) -> None:
        """Fit one BDT per (time step, channel) into `bdt_grid`."""
        if sample_weights is None:      # Uniform weights == unweighted fit
            sample_weights = np.ones(len(y))

        ensemble = []
        with tqdm(total = self.n_steps * self.n_channels,
                  desc  = "Training BDT grid") as progress_bar:
            for step in range(self.n_steps):
                xt = self._inputs(x[step], conditions)

                for channel in range(self.n_channels):
                                        # Only give samples where this slot is real
                    real = slice(None) if mask is None else mask[:, channel]
                    ensemble.append(self._fit_one(xt[real], y[real, channel], sample_weights[real]))
                    progress_bar.update()

        self.bdt_grid = (np.array(ensemble, dtype = object)
                           .reshape(self.n_steps, self.n_channels))


    def predict(self, t: float, xt: NDArray, conditions: NDArray | None = None) -> NDArray:
        """Velocity at time `t` at `xt`, both (N, n_channels)."""
        step = t_to_step(t, self.n_steps)

                                        # Unconditioned
        if self.max_multiplicity is None and conditions is None:
            return np.column_stack([ bdt.predict(xt) for bdt in self.bdt_grid[step] ])

        if self.max_multiplicity is None or conditions is None:
            raise ValueError("Pass conditions when and only when the FlowBDT was "
                             "trained conditioned on them.")

        inputs   = self._inputs(xt, conditions)
        velocity = np.zeros_like(xt)    # Keep padding at zero velocity
        mask     = slot_mask(conditions[:, 0], self.n_channels, self.max_multiplicity)

        for channel, bdt in enumerate(self.bdt_grid[step]):
            real = mask[:, channel]
            velocity[real, channel] = self._predict_one(bdt, inputs[real],
                                                        channel % self.max_multiplicity)

        return velocity


    def sample_multiplicities(self, n_samples: int,
                              rng: np.random.Generator | None = None) -> NDArray | None:
        """Draw multiplicities from the training cut's histogram as `(n_samples, 1)`
        conditions, `None` for a FlowBDT not conditioned on multiplicity alone.
        """
        if self.multiplicity_prior is None:
            return None

        rng = rng or self.rng or np.random.default_rng()
        prior = self.multiplicity_prior
        return rng.choice(len(prior), (n_samples, 1), p = prior)


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
        vector field to generate a new sample. If conditioned, `conditions` are
        drawn with `sample_multiplicities` unless given, and appended as trailing
        columns.
        """
        shape = None if n_samples is None else (n_samples, self.n_channels)
        x0    = initial_noise(shape, x0, rng or self.rng)

        if conditions is None:
            conditions = self.sample_multiplicities(len(x0), rng)

        conditioned_predict = partial(self.predict, conditions = conditions)
        raw = solver(conditioned_predict, x0, self.n_steps)

        if conditions is None:
            return raw

        return np.column_stack([ raw, conditions.astype(raw.dtype) ])

    def reflow(self, n_steps: int, n_pairs: int) -> FlowBDT:
        """Reflow this FlowBDT: train a FlowBDT with `n_steps` on `n_pairs` of 
        noise and generated samples from it.
        
        (2-rectified flow, arXiv:2209.03003).
        """
        x0 = initial_noise((n_pairs, self.n_channels), rng = self.rng)
        x1 = self.sample(x0 = x0)

        conditions = None               # Split off the trailing conditions
        if self.max_multiplicity is not None:
            conditions = x1[:, self.n_channels:]
            x1 = x1[:, :self.n_channels]

        reflowed = FlowBDT(self.tree_config, self.rng)
        reflowed.fit(*build_trainds(x1, n_steps, x0, conditions = conditions,
                                    max_multiplicity = self.max_multiplicity))

        return reflowed


# --- Export/import ---

    def save(self, path: str) -> None:
        """Persist the whole state except for `rng`, and export boosters to 
        binary JSON.
        """
        state = { key: value for key, value in self.__dict__.items() if key != "rng" }
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
