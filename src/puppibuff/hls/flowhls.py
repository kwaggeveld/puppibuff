from __future__ import annotations

from ..flowbdt import FlowBDT
from ..utils import initial_noise, midpoint_solve, t_to_step
from . import write
from .compile import compile_grid, compile_flowhls
from .convert import convert_grid
from .load import attach_bridge, import_bridge, load_grid
from .utils import FLOWHLS_PROJECT, bridge_path

from pathlib import Path

from conifer.backends import xilinxhls
from conifer.utils.performance import performance_estimates
import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def _as_flat_f64(x: NDArray) -> NDArray:
    """The bridge's `std::vector<double>` argument."""
    return np.asarray(x, dtype = np.float64).ravel()


class FlowHLS:
    """A FlowBDT whose velocity fields are evaluated by conifer HLS models
    instead of XGBoost. `predict`/`sample` methods mirror FlowBDT's.

    The grid is written as a single project holding every BDT, which is what lets
    the sampling itself run on the FPGA. `per_bdt = True` writes conifer's own
    layout instead, one project per BDT.
    """
    
    accum_precision = "ap_fixed<32,8>"  # The solver's state, wider than score_t

    flowhls_top = write.SAMPLE_TOP


# --- Constructors ---

    def __init__(self, grid: NDArray, output_dir: str = "flowhls") -> None:
        self.bdt_grid   = grid          # (n_steps, n_groups) grid of conifer models
        self.output_dir = Path(output_dir).resolve()
        self.n_steps    = grid.shape[0]
        self.n_channels = grid[0, 0].n_features

        self.per_bdt = False
        self._bridge = None


    @classmethod
    def convert(
        cls,
        model: FlowBDT,
        config: dict | None = None,     # None => hls_config()
        output_dir: str = "flowhls",
    ) -> FlowHLS:
        return cls(convert_grid(model, config, output_dir), output_dir)


    @classmethod
    def load(cls, work_dir: str = "flowhls", per_bdt: bool = False) -> FlowHLS:
        """Reuse a grid already converted and compiled into `work_dir`, binding
        whichever bridge `compile` built there. Reloading is the only way to reuse
        a merged build, since it compiles at most once per process.
        """
        hls = cls(load_grid(work_dir, attach = per_bdt), work_dir)

        if not per_bdt:
            hls.bridge = import_bridge(bridge_path(hls.output_dir, FLOWHLS_PROJECT))

        hls.per_bdt = per_bdt

        return hls


# --- BDT grid ---

    @property
    def grouped_names(self) -> list[list[str]]:
        """The BDT project names, nested by step. Shape `write.flowhls_cpp` needs."""
        return [ [ model.config.project_name for model in row ]
                 for row in self.bdt_grid ]


    def _call_on_grid(self, method: str, **kwargs) -> list:
        """Call `method` on every conifer model in `self.bdt_grid`."""
        models_pbar = tqdm(self.bdt_grid.flat, total = self.bdt_grid.size, desc = method)
        return [ getattr(model, method)(**kwargs) for model in models_pbar ]


# --- Writing ---

    def write(self, per_bdt: bool = False) -> None:
        """Write the grid's HLS project(s): one full design, or one per BDT."""
        if per_bdt:
            self._call_on_grid("write")
        else:
            self._write_flowhls()


    def _write_flowhls(self) -> None:
        ref = self.bdt_grid[0, 0]

        if not ref.config.unroll:       # The rolled variant reads its trees from
            raise NotImplementedError(  # an array `parameters.h` we do not write
                "Merging is implemented for Unroll = True only"
            )

        bdt_h = (Path(xilinxhls.__file__).parent
                 / "firmware" / "BDT_unrolled.h").read_text()

        files_to_write = {
            "firmware/BDT.h":           write.bdt_h_patch(bdt_h),
            "firmware/ap_types.h":      write.ap_types_h(self.n_channels, ref.config, self.accum_precision,),
            "firmware/flowhls.h":       write.flowhls_h(self.n_steps),
            "firmware/flowhls.cpp":     write.flowhls_cpp(self.grouped_names),
            "bridge.cpp":               write.bridge_cpp(FLOWHLS_PROJECT, self.n_steps),
            "hls_parameters.tcl":       write.hls_parameters_tcl(FLOWHLS_PROJECT, self.flowhls_top, ref.config.xilinx_part, ref.config.clock_period),
            "build_hls.tcl":            write.build_hls_tcl(),
        } | {                           # One header per BDT
            f"firmware/{name}.h":       write.tree_header(model, name)
            for model in self.bdt_grid.flat
            for name in [ model.config.project_name ]
        }
                                        # Create firmware directory
        (self.output_dir / "firmware").mkdir(parents = True, exist_ok = True)

        for file, source in files_to_write.items():
            (self.output_dir / file).write_text(source)

        self._call_on_grid("save")      # Write all .json files for `cls.load`


# --- Compiling and building ---

    def compile(self, n_threads: int | None = None, per_bdt: bool = False) -> None:
        """Compile for emulation. Conifer implements Python bindings for the HLS
        code. The merged design is one compilation unit, so `n_threads` applies
        only to the per-BDT grid (None => all cores).
        """
        self.write(per_bdt = per_bdt)

        if per_bdt:
            compile_grid(self.bdt_grid, n_threads)

            for model in self.bdt_grid.flat:    # A bridge only exists in the process
                attach_bridge(model)            # opened it, so re-attach in parent

        else:
            compile_flowhls(self.output_dir, FLOWHLS_PROJECT)
            self.bridge = import_bridge(bridge_path(self.output_dir, FLOWHLS_PROJECT))

        self.per_bdt = per_bdt
        

    def build(self, per_bdt: bool = True, **kwargs) -> list[bool]:
        """Run HLS synthesis over the per-BDT projects. Needs vitis_hls on PATH."""
        if not per_bdt:
            raise NotImplementedError(
                "Synthesise the merged project with "
                f"`vitis_hls -f build_hls.tcl` in {self.output_dir}"
            )

        return self._call_on_grid("build", **kwargs)


# --- Sampling ---

    @property
    def bridge(self):
        """The merged design's pybind11 bridge, bound by `compile()` or `load()`."""
        if self._bridge is None:
            raise RuntimeError("No bridge. Call compile() or load().")

        return self._bridge

    @bridge.setter
    def bridge(self, bridge) -> None:
        self._bridge = bridge


    def predict(self, t: float, xt: NDArray) -> NDArray:
        # xt has shape (N, n_channels)
        step = t_to_step(t, self.n_steps)

        if self.per_bdt:
            return np.column_stack(     # (N, n_channels)
                [ model.decision_function(xt) for model in self.bdt_grid[step] ]
            )

        return (np.array(self.bridge.field(step, _as_flat_f64(xt)))
                    .reshape(xt.shape))


    def sample(
        self,
        n_samples: int | None = None,
        x0: NDArray | None = None
    ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new event.
        """
        x0 = initial_noise(n_samples, self.n_channels, x0)

        if self.per_bdt:
            return midpoint_solve(self.predict, x0, self.n_steps)

        return (np.array(self.bridge.sample(_as_flat_f64(x0)))
                    .reshape(x0.shape))


# --- Resources ---

    def resource_estimates(self) -> dict[str, int]:
        """conifer's LUT/FF estimates for the converted grid. These describe the
        BDTs, not the complete FlowHLS project.
        """
        estimates = [ performance_estimates(model) for model in self.bdt_grid.flat ]

        return {
            "lut": sum(estimate["lut"] for estimate in estimates),
            "ff":  sum(estimate["ff"]  for estimate in estimates),
        }
