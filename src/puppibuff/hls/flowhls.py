from __future__ import annotations

from ..flowbdt import FlowBDT
from ..solvers import ab2_solve, Solver
from ..utils import initial_noise, t_to_step
from . import write
from .compile import compile_grid, compile_flowhls
from .convert import convert_grid
from .load import attach_bridge, import_bridge, load_grid
from .utils import BDT_DATA, merged_bridge, merged_build, project_paths

from pathlib import Path
from conifer.utils.performance import performance_estimates
import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray
from typing import Iterator

#-----------------------------------------------------------------------------

def _as_flat_f64(x: NDArray) -> NDArray:
    """The bridge's `std::vector<double>` argument."""
    return np.asarray(x, dtype = np.float64).ravel()


class FlowHLS:
    """A FlowBDT whose velocity fields are evaluated by conifer HLS models
    instead of XGBoost. `predict`/`sample` methods mirror FlowBDT's.

    The grid is written as a single `merged` project holding every BDT, which is
    what lets the sampling itself run on the FPGA. `merged = False` writes conifer's
    own layout instead, one project per BDT.
    """

    accum_precision = "ap_fixed<32,8>"  # The solver's state, wider than score_t


# --- Constructors ---

    def __init__(
        self,
        grid: NDArray,
        output_dir: Path | str = "flowhls",
        merged: bool = True,
    ) -> None:
        self.bdt_grid   = grid          # (n_steps, n_groups) grid of conifer models
        self.output_dir = Path(output_dir).resolve()
        self.n_steps    = grid.shape[0]
        self.n_channels = grid[0, 0].n_features

        self.merged  = merged
        self._bridge = None


    @classmethod
    def convert(
        cls,
        model: FlowBDT,
        config_overrides: dict | None = None,
        output_dir: str = "flowhls",
        merged: bool = True,
    ) -> FlowHLS:
        return cls(convert_grid(model, config_overrides, output_dir), output_dir, merged)


    @classmethod
    def load(cls, work_dir: str = "flowhls") -> FlowHLS:
        """Reuse a grid already converted and compiled into `work_dir`, binding
        whichever bridge `compile` built there.
        """
        root   = Path(work_dir).resolve()
        merged = merged_build(root)
                                        # The merged design keeps its BDTs under
                                        # `bdt_data`, and shares one bridge
        grid = load_grid(root / BDT_DATA if merged else root, attach = not merged)

        hls = cls(grid, root, merged)

        if merged:
            hls.bridge = import_bridge(merged_bridge(root))

        return hls


# --- BDT grid ---

    @property
    def grouped_names(self) -> list[list[str]]:
        """The BDT project names, nested by step. Shape `write.field_sXX_cpp` needs."""
        return [ [ model.config.project_name for model in row ]
                 for row in self.bdt_grid ]


    @property
    def blocks(self) -> list[str]:
        """Every HLS project of the merged design. `narrow` is inlined into 
        its callers and the sampler is left to the VHDL, so neither is a block.
        """
        return ([ write.field_name(step) for step in range(self.n_steps) ]
                + [ write.STEP_TOP ])


    @property
    def cpp_sources(self) -> list[str]:
        """The design's translation units, for `compile`. The blocks' own, plus
        the two that only the emulated design compiles: `narrow` is inlined by
        HLS and `sample` is the VHDL top's C++ reference.
        """
        return ([ f"firmware/{ block }.cpp" for block in self.blocks ]
                + [ "firmware/narrow.cpp", "firmware/sample.cpp" ])


    def _call_on_grid(self, method: str, **kwargs) -> list:
        """Call `method` on every conifer model in `self.bdt_grid`."""
        models_pbar = tqdm(self.bdt_grid.flat, total = self.bdt_grid.size, desc = method)
        return [ getattr(model, method)(**kwargs) for model in models_pbar ]


# --- Writing ---

    def write(self) -> None:
        """Write the grid's HLS project(s): one full design, or one per BDT."""
        if self.merged:
            self._write_flowhls()
        else:
            self._call_on_grid("write")


    def _hls_project_files(self, cfg) -> Iterator[tuple[str, str]]:
        """Construct the `.tcl` file and source of each block's HLS project."""
        for block in self.blocks:
            yield f"blocks/{ block }/hls_parameters.tcl", write.hls_parameters_tcl(block, cfg.xilinx_part, cfg.clock_period)
            yield f"blocks/{ block }/build_hls.tcl",      write.build_hls_tcl()
            yield f"blocks/{ block }/vivado_synth.tcl",   write.vivado_synth_tcl(block, cfg.xilinx_part)


    def _save_bdt_data(self) -> None:
        """Save every BDT's `.json`, in the same step/group tree 
        `merged = False` writes.
        """
        models_pbar = tqdm(np.ndenumerate(self.bdt_grid),
                           total = self.bdt_grid.size, desc = "save")

        for (step, group), model in models_pbar:
            project_dir, name = project_paths(self.output_dir / "bdt_data", step, group)
            model.save(str(project_dir / f"{name}.json"))


    def _write_flowhls(self) -> None:
        bdt_config = self.bdt_grid[0, 0].config

        files_to_write = {
            "firmware/bdt_grid/BDT.h":              write.bdt_h(bdt_config.unroll),
            "firmware/ap_types.h":                  write.ap_types_h(self.n_channels, bdt_config, self.accum_precision,),
            "firmware/flowhls.h":                   write.flowhls_h(self.n_steps),
            f"firmware/{ write.STEP_TOP }.cpp":     write.ab2_step_cpp(self.n_steps),
            "firmware/narrow.cpp":                  write.narrow_cpp(),
            "firmware/sample.cpp":                  write.sample_cpp(self.n_steps),
            "bridge.cpp":                           write.bridge_cpp(self.n_steps),
            "build_all.sh":                         write.build_all_sh(self.blocks),

            **{                         # One source per flow step
                f"firmware/{ write.field_name(step) }.cpp": write.field_sXX_cpp(step, bdts_in_step)
                for step, bdts_in_step in enumerate(self.grouped_names)
            },

            **{                         # One header per BDT
                f"firmware/bdt_grid/{ model.config.project_name }.h": write.bdt_sXX_gXX_h(model)
                for model in self.bdt_grid.flat
            },
                                        # HLS project `.tcl` scripts
            **dict(self._hls_project_files(bdt_config))
        }

        for file, source in files_to_write.items():
            path = self.output_dir / file
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(source)

                                        # Make script executable
        (self.output_dir / "build_all.sh").chmod(0o755)

        self._save_bdt_data()           # Write all .json files for `cls.load`


# --- Compiling and building ---

    def compile(self, n_threads: int | None = None) -> None:
        """Compile for emulation. Conifer implements Python bindings for the HLS
        code. The merged design is one compilation, so `n_threads` applies
        only to the per-BDT grid (None => all cores).
        """
        self.write()

        if self.merged:
            compile_flowhls(self.output_dir, self.cpp_sources)
            self.bridge = import_bridge(merged_bridge(self.output_dir))
        else:
            compile_grid(self.bdt_grid, n_threads)

            for model in self.bdt_grid.flat:    # A bridge only exists in the process
                attach_bridge(model)            # opened it, so re-attach in parent


    def build(self, **kwargs) -> list[bool]:
        """Run HLS synthesis over the per-BDT projects. Needs vitis_hls on PATH."""
        if self.merged:                 # Every block is its own project, so
            raise NotImplementedError(  # synthesis is a loop rather than a call
                f"Synthesise the merged design with `./build_all.sh` "
                f"in {self.output_dir}"
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

        if self.merged:
            return (np.array(self.bridge.field(step, _as_flat_f64(xt)))
                        .reshape(xt.shape))

        return np.column_stack(         # (N, n_channels)
            [ model.decision_function(xt) for model in self.bdt_grid[step] ]
        )


    def sample(
        self,
        n_samples: int | None = None,
        x0: NDArray | None = None,
        solver: Solver = ab2_solve,
    ) -> NDArray:
        """Starting from noise, provided or sampled here, integrate the learnt
        vector field to generate a new event.

        The per-BDT layout integrates in Python and so takes any solver. The
        merged design only supports `midpoint_solve` integration for now.
        """
        x0 = initial_noise(n_samples, self.n_channels, x0)

        if self.merged:
            if solver is not write.SAMPLE_SOLVER:
                raise NotImplementedError(
                    f"This merged design has {write.SAMPLE_SOLVER.__name__} "
                    f"compiled into it, so it cannot sample with "
                    f"{solver.__name__}: pass that solver instead or use "
                    f"merged = False."
                )

            return (np.array(self.bridge.sample(_as_flat_f64(x0)))
                        .reshape(x0.shape))

        return solver(self.predict, x0, self.n_steps)


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
