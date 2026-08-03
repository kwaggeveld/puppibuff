from __future__ import annotations

from ..flowbdt import FlowBDT
from ..utils import midpoint_solve, t_to_step
from .compile import compile_grid
from .convert import convert_grid
from .load import attach_bridge, load_grid

from conifer.utils.performance import performance_estimates
import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlowHLS:
    """A FlowBDT whose velocity fields are evaluated by conifer HLS models
    instead of XGBoost. `predict`/`sample` methods mirror FlowBDT's.
    """

    def __init__(self, grid: NDArray) -> None:
        self.bdt_grid   = grid          # (n_steps, n_groups) grid of conifer models
        self.n_steps    = grid.shape[0]
        self.n_channels = grid[0, 0].n_features


    @classmethod
    def convert(
        cls,
        model: FlowBDT,
        config: dict | None = None,     # None => hls_config()
        output_dir: str = "hls",
    ) -> FlowHLS:
        return cls(convert_grid(model, config, output_dir))


    @classmethod
    def load(cls, output_dir: str = "hls") -> FlowHLS:
        """Reuse a grid already converted and compiled into `output_dir`."""
        return cls(load_grid(output_dir))


    def _call_on_grid(self, method: str, **kwargs) -> list:
        """Call `method` on every conifer model in `self.bdt_grid`."""
        models_pbar = tqdm(self.bdt_grid.flat, total = self.bdt_grid.size, desc = method)
        return [ getattr(model, method)(**kwargs) for model in models_pbar ]


    def write(self) -> None:
        """Write every BDT's HLS project to its output directory."""
        self._call_on_grid("write")


    def compile(self, n_threads: int | None = None) -> None:
        """Compile every BDT for emulation, `n_threads` at a time (None => all
        cores). Conifer implements Python bindings for the HLS code."""

        self.write()                           # Write here instead of in the threads,
        compile_grid(self.bdt_grid, n_threads) # so each has a project JSON to load
            
        for model in self.bdt_grid.flat:    # A bridge only exists in the process
            attach_bridge(model)            # opened it, so re-attach in parent


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
         
        return np.column_stack(             # (N, n_channels)
            [ model.decision_function(xt) for model in self.bdt_grid[step] ]
        )


    def sample(self, n_samples: int) -> NDArray:
        x0 = np.random.normal(size = (n_samples, self.n_channels)).astype(np.float32)
        return midpoint_solve(self.predict, x0, self.n_steps)
