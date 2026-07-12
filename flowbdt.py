from .solvers import midpoint_solve

from xgboost import XGBRegressor, XGBModel
from joblib import Parallel, delayed    # type: ignore[import-untyped]
import numpy as np

from numpy.typing import NDArray

from tqdm_joblib import ParallelPbar



class FlowBDT():
    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def _fit_one(self, x: NDArray, y: NDArray) -> XGBModel:
        return XGBRegressor(**self.config).fit(x, y)

    def fit(self, xt: NDArray, yt: NDArray, n_jobs: int = 1) -> None:
        # xt, yt have shape `(n_steps, N, n_channels)`

        self.n_steps, _, self.n_channels = xt.shape

        jobs = (                        # One BDT per (step, channel)
            delayed(self._fit_one)(xt[step], yt[step][:, channel])
            for step    in range(self.n_steps)
            for channel in range(self.n_channels)
        )

        ensemble = ParallelPbar("Training BDT grid")(n_jobs = n_jobs)(jobs)
        self.bdt_grid = np.array(ensemble, dtype = object)
        self.bdt_grid = self.bdt_grid.reshape(self.n_steps, self.n_channels)

    def predict(self, t: float, xt: NDArray) -> NDArray:
        # xt has shape (N, n_channels)
                                        # Clip t * (n_steps - 1) to
                                        # [0, n_steps - 1]
        step = max(0, min(round(t * (self.n_steps - 1)), self.n_steps - 1))

        ret = np.empty_like(xt)         # Call approp. BDT for each channel
        for channel in range(self.n_channels):
            ret[:, channel] = self.bdt_grid[step, channel].predict(xt)
        return ret                      # (N, n_channels)

    def sample(self, n_samples: int) -> NDArray:
        x0 = np.random.normal(size = (n_samples, self.n_channels)).astype(np.float32)
        return midpoint_solve(self.predict, x0, self.n_steps)                 # type: ignore[reportArgumentType]