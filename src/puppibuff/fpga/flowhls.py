from __future__ import annotations

from ..utils import FlowBDT
from ..solvers import midpoint_solve, t_to_step
from .export import convert_grid

from conifer.utils.performance import performance_estimates
import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlowHLS:
    """A FlowBDT whose velocity fields are evaluated by conifer HLS models
    instead of XGBoost. `predict`/`sample` mirror FlowBDT's, so a decoded HLS
    sample goes straight into the same plots as a decoded Python one.
    """

    def __init__(self, grid: NDArray, n_channels: int) -> None:
        self.bdt_grid   = grid          # (n_steps, n_groups) of conifer models
        self.n_steps    = grid.shape[0]
        self.n_channels = n_channels


    @classmethod
    def convert(
        cls,
        model: FlowBDT,
        config: dict | None = None,     # None => hls_config()
        output_dir: str = "hls",
    ) -> FlowHLS:
        return cls(convert_grid(model, config, output_dir), model.n_channels)

    def _call_on_grid(self, method: str, **kwargs) -> list:
        """Call `method` on every conifer model in `self.grid`."""
        models_pbar = tqdm(self.bdt_grid.flat, total = self.bdt_grid.size, desc = method)
        return [getattr(model, method)(**kwargs) for model in models_pbar]


    def write(self) -> None:
        """Write every BDT's HLS project to its output directory."""
        self._call_on_grid("write")


    def compile(self) -> None:
        """Compile every BDT for emulation. Conifer implements Python
        bindings for the HLS code."""
        self._call_on_grid("compile")


    def build(self, **kwargs) -> list[bool]:
        """Run HLS synthesis on every BDT. Needs vitis_hls on PATH."""
        return self._call_on_grid("build", **kwargs)


    def resource_estimates(self) -> dict[str, int]:
        """conifer's LUT/FF/latency estimates for the converted grid."""
        estimates = [ performance_estimates(model) for model in self.bdt_grid.flat ]

        return {
            "lut": sum(estimate["lut"] for estimate in estimates),
            "ff":  sum(estimate["ff"]  for estimate in estimates),
        }


    def predict(self, t: float, xt: NDArray) -> NDArray:
        # xt has shape (N, n_channels)
        step = t_to_step(t, self.n_steps)

                                        # (N, n_channels)
        return np.column_stack(
            [ model.decision_function(xt) for model in self.bdt_grid[step] ]
        )


    def sample(self, n_samples: int) -> NDArray:
        x0 = np.random.normal(size = (n_samples, self.n_channels)).astype(np.float32)
        return midpoint_solve(self.predict, x0, self.n_steps)
