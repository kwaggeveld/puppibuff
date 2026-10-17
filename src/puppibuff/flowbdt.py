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


    def _inputs(self, xt: NDArray, multiplicities: NDArray | None) -> NDArray:
        """`xt` with the `multiplicities` appended if given."""
        if multiplicities is None:
            return xt

        return np.column_stack([ xt, multiplicities ]).astype(np.float32)


    def fit(
        self,
        x: Paths,
        y: NDArray,
        sample_weights: NDArray | None = None, # (N,), one weight per event
    ) -> None:
        # x: n_steps points along the paths, each (N, n_channels); y: (N, n_channels)
        self.n_steps    = x.n_steps
        self.n_channels = y.shape[1]

        multiplicities: NDArray | None = None
        mask: NDArray | None = None
        if x.real is not None:          # Constituents: rebuild Paths
            self.max_multiplicity = x.real.shape[1]
            multiplicities = x.real.sum(axis = 1)
            
            mask = slot_mask(multiplicities, self.n_channels, self.max_multiplicity)      # pyright: ignore[reportArgumentType]

            counts = np.bincount(multiplicities, minlength = self.max_multiplicity + 1)   # pyright: ignore[reportOptionalOperand]
            self.multiplicity_prior = counts / counts.sum()
                                        # Padding is paired with its own noise
            x = Paths(x.x0, np.where(mask, x.x1, x.x0), x.ts, x.real)

        self._train_grid(x, y, multiplicities, mask, sample_weights)


    def _train_grid(
        self,
        x: Paths,
        y: NDArray,
        multiplicities: NDArray | None, # (N,), if conditioned
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
                xt = self._inputs(x[step], multiplicities)

                for channel in range(self.n_channels):
                                        # Only give samples where this slot is real
                    real = slice(None) if mask is None else mask[:, channel]
                    ensemble.append(self._fit_one(xt[real], y[real, channel], sample_weights[real]))
                    progress_bar.update()

        self.bdt_grid = (np.array(ensemble, dtype = object)
                           .reshape(self.n_steps, self.n_channels))


    def predict(self, t: float, xt: NDArray, multiplicities: NDArray | None = None) -> NDArray:
        """Velocity at time `t` at `xt`, both (N, n_channels)."""
        step = t_to_step(t, self.n_steps)

                                        # Unconditioned
        if self.max_multiplicity is None and multiplicities is None:
            return np.column_stack([ bdt.predict(xt) for bdt in self.bdt_grid[step] ])

        if self.max_multiplicity is None or multiplicities is None:
            raise ValueError("Pass multiplicities when and only when the FlowBDT was "
                             "trained conditioned on them.")

        inputs   = self._inputs(xt, multiplicities)
        velocity = np.zeros_like(xt)    # Keep padding at zero velocity
        mask     = slot_mask(multiplicities, self.n_channels, self.max_multiplicity)

        for channel, bdt in enumerate(self.bdt_grid[step]):
            real = mask[:, channel]
            velocity[real, channel] = self._predict_one(bdt, inputs[real],
                                                        channel % self.max_multiplicity)

        return velocity


    def sample_multiplicities(self, n_samples: int,
                              rng: np.random.Generator | None = None) -> NDArray | None:
        """Draw multiplicities from the training cut's histogram, `None` for a
        FlowBDT not conditioned on multiplicity.
        """
        if self.multiplicity_prior is None:
            return None

        rng = rng or self.rng or np.random.default_rng()
        prior = self.multiplicity_prior
        return rng.choice(len(prior), n_samples, p = prior)


    def sample(
        self,
        n_samples: int | None = None,
        x0: NDArray | None = None,
        solver: Solver = ab2_solve,
        rng: np.random.Generator | None = None,
        *,
        multiplicities: NDArray | None = None,
    ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new sample, conditioned on `multiplicities`
        if given, or sampled here if needed.
        """
        shape = None if n_samples is None else (n_samples, self.n_channels)
        x0    = initial_noise(shape, x0, rng or self.rng)

        if multiplicities is None:
            multiplicities = self.sample_multiplicities(len(x0), rng)

        conditioned_predict = partial(self.predict, multiplicities = multiplicities)
        raw = solver(conditioned_predict, x0, self.n_steps)

        if multiplicities is None:
            return raw

        return np.column_stack([ raw, multiplicities.astype(raw.dtype) ])


    def reflow(self, n_steps: int, n_pairs: int) -> FlowBDT:
        """Reflow this FlowBDT: train a FlowBDT with `n_steps` on `n_pairs` of 
        noise and generated samples from it.
        
        (2-rectified flow, arXiv:2209.03003).
        """
        x0 = initial_noise((n_pairs, self.n_channels), rng = self.rng)
        x1 = self.sample(x0 = x0)

        real = None                     # Split off the trailing multiplicities
        if self.max_multiplicity is not None:
            x1, real = x1[:, :-1], np.arange(self.max_multiplicity) < x1[:, -1:]

        reflowed = FlowBDT(self.tree_config, self.rng)
        reflowed.fit(*build_trainds(x1, n_steps, x0, real = real))

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
