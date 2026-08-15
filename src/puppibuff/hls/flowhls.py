from __future__ import annotations

from ..codecs import Codec
from ..flowbdt import FlowBDT
from ..solvers import ab2_solve, Solver
from ..utils import initial_noise, t_to_step
from . import constants as c
from . import write
from .compile import compile_grid, compile_flowhls
from .convert import convert_grid
from .load import attach_bridge, import_bridge, load_grid
from .utils import block_latency, merged_bridge, merged_build, project_paths

import subprocess
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
        self._codec: Codec | None  = None


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
        grid = load_grid(root, attach = not merged)

        hls = cls(grid, root, merged)

        if merged:
            hls.bridge = import_bridge(merged_bridge(root))

        return hls


    @property
    def codec(self) -> Codec:
        """The codec `write` was given. Read back off the design when this object
        is not the one that wrote it, so a converted or loaded grid can decode
        and write its payload without being handed the codec a second time.
        """
        if self._codec is None:
            saved = self.output_dir / c.CODEC_FILE

            if not saved.exists():
                raise FileNotFoundError(
                    f"No codec in { self.output_dir }. Call write(codec) first."
                )

            self._codec = Codec.from_json(saved)

        return self._codec


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
        return ([ c.FIELD_NAME(step) for step in range(self.n_steps) ]
                + [ c.STEP_TOP, Codec.s_DECODE_TOP ])


    @property
    def cpp_sources(self) -> list[str]:
        """The design's translation units, for `compile`. The blocks' own, plus
        `sample` for emulation.
        """
        return ([ f"firmware/{ block }.cpp" for block in self.blocks ]
                + [ "firmware/sample.cpp" ])


    def _call_on_grid(self, method: str, **kwargs) -> list:
        """Call `method` on every conifer model in `self.bdt_grid`."""
        models_pbar = tqdm(self.bdt_grid.flat, total = self.bdt_grid.size, desc = method)
        return [ getattr(model, method)(**kwargs) for model in models_pbar ]


# --- Writing ---

    def write(self, codec: Codec) -> None:
        """Write the grid's HLS project(s): one full design, or one per BDT.
        The `codec` writes the design's `decode`.
        """
        self._codec = codec

        self.output_dir.mkdir(parents = True, exist_ok = True)
        codec.to_json(self.output_dir / c.CODEC_FILE)

        if self.merged:
            self._write_flowhls()
        else:
            self._call_on_grid("write")


    def _hls_project_files(self, cfg) -> Iterator[tuple[str, str]]:
        """Construct the `.tcl` file and source of each block's HLS project."""
        for block in self.blocks:
            yield f"{ c.BLOCKS_DIR }/{ block }/hls_parameters.tcl", write.hls_parameters_tcl(block, cfg.xilinx_part, cfg.clock_period)
            yield f"{ c.BLOCKS_DIR }/{ block }/build_hls.tcl",      write.build_hls_tcl()
            yield f"{ c.BLOCKS_DIR }/{ block }/vivado_synth.tcl",   write.vivado_synth_tcl(block, cfg.xilinx_part)


    def _save_bdt_data(self) -> None:
        """Save every BDT's `.json`, in the same step/group tree 
        `merged = False` writes.
        """
        models_pbar = tqdm(np.ndenumerate(self.bdt_grid),
                           total = self.bdt_grid.size, desc = "save")

        for (step, group), model in models_pbar:
            project_dir, name = project_paths(self.output_dir, step, group)
            model.save(str(project_dir / f"{name}.json"))


    def _write_flowhls(self) -> None:
        bdt_config = self.bdt_grid[0, 0].config

        files_to_write = {
            "firmware/bdt_grid/BDT.h":              write.bdt_h(bdt_config.unroll),
            "firmware/ap_types.h":                  write.ap_types_h(self.n_channels, bdt_config, c.ACCUM_PRECISION, self.codec),
            "firmware/flowhls.h":                   write.flowhls_h(self.n_steps),
            f"firmware/{ c.STEP_TOP }.cpp":         write.ab2_step_cpp(self.n_steps),
            f"firmware/{ Codec.s_DECODE_TOP }.cpp": self.codec.decode_cpp(),
            f"firmware/{ Codec.s_DECODE_PARAMS }":  self.codec.decode_params_h(),
            "firmware/sample.cpp":                  write.sample_cpp(self.n_steps),
            "bridge.cpp":                           write.bridge_cpp(self.n_steps),
            c.BUILD_SCRIPT:                         write.build_all_sh(self.blocks),

            **{                         # One source per flow step
                f"firmware/{ c.FIELD_NAME(step) }.cpp": write.field_sXX_cpp(step, bdts_in_step)
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
        (self.output_dir / c.BUILD_SCRIPT).chmod(0o755)

        self._save_bdt_data()           # Write all .json files for `cls.load`


# --- Compiling and building ---

    def compile(self, n_threads: int | None = None) -> None:
        """Compile the written sources for emulation. Conifer implements Python
        bindings for the HLS code. The merged design is one compilation, so
        `n_threads` applies only to the per-BDT grid (None => all cores).
        """
        if self.merged:
            compile_flowhls(self.output_dir, self.cpp_sources)
            self.bridge = import_bridge(merged_bridge(self.output_dir))
        else:
            compile_grid(self.bdt_grid, n_threads)

            for model in self.bdt_grid.flat:    # A bridge only exists in the process
                attach_bridge(model)            # opened it, so re-attach in parent


    def build(self, *options: str) -> None:
        """Synthesise the written design. Needs vitis_hls on PATH.

        If a merged design, call `build_all.sh` written by `write()`. Then
        `options` pass through to each block's `build_hls.tcl`  If a per-
        BDT design, call `build` on each conifer model.
        """
        if not self.merged:
            self._call_on_grid("build")
            return

        if not (self.output_dir / c.BUILD_SCRIPT).exists():
            raise FileNotFoundError(
                f"Nothing to build in { self.output_dir }, { c.BUILD_SCRIPT } "
                f"is missing. Call `FlowHLS.write(codec)` first."
            )

        subprocess.run([ f"./{ c.BUILD_SCRIPT }", *options ],
                       cwd = self.output_dir, check = True)


# --- EMP payload ---

    @property
    def latencies(self) -> dict[str, int]:
        """Every block's scheduled latency, in clock cycles."""
        return { block: block_latency(self.output_dir, block)
                 for block in self.blocks }


    def write_payload(self) -> None:
        """Write the payload that ties the synthesised blocks together.

        Separate from `write` as block latencies are required to construct the
        script.
        """
        if not self.merged:             # The per-BDT layout has no blocks to 
            raise NotImplementedError(  # tie together
                "The payload combines a merged design's blocks."
            )

        (self.output_dir / c.PAYLOAD_FILE).write_text(
            write.emp_payload_vhd(
                self.n_steps, self.n_channels, self.latencies,
                self.bdt_grid[0, 0].config, c.ACCUM_PRECISION, self.codec,
            )
        )


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
            if solver is not c.SAMPLE_SOLVER:
                raise NotImplementedError(
                    f"This merged design has {c.SAMPLE_SOLVER.__name__} "
                    f"compiled into it, so it cannot sample with "
                    f"{solver.__name__}: pass that solver instead or use "
                    f"merged = False."
                )

            return (np.array(self.bridge.sample(_as_flat_f64(x0)))
                        .reshape(x0.shape))

        return solver(self.predict, x0, self.n_steps)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
        """Decode a sample through the design's own `decode` HLS block."""
        if not self.merged:             # The per-BDT layout is BDTs only, with
            raise NotImplementedError(  # no decode block to run
                "A per-BDT layout has no `decode` HLS block, "
                "so decode with the `Codec` itself."
            )

        codec = self.codec
                                        # One channel-major row per event
        decoded = (np.array(self.bridge.decode(_as_flat_f64(out)))
                       .reshape(len(out), len(codec.s_DECODED), codec.multiplicity))

        if codec.multiplicity == 1:     # If flat data, drop the slot axis
            decoded = decoded[..., 0]

        return dict(zip(codec.s_DECODED, np.moveaxis(decoded, 1, 0)))


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
