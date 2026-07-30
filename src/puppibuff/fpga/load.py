from __future__ import annotations

from .utils import project_paths

import importlib.util
from pathlib import Path
import numpy as np
from conifer.model import ModelBase, load_model

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def attach_bridge(model: ModelBase) -> None:
    """Bind an already-compiled pybind11 bridge to `model`.

    This is roughly the tail of conifer's `XilinxHLSModel.compile()`, 
    which always rewrites and re-compiles.
    """
    config = model.config

    bridge = Path(config.output_dir) / f"conifer_bridge_{config.project_name}.so"   # type: ignore
    if not bridge.exists():
        raise FileNotFoundError(f"{bridge} not found, compile it first")

                                        # For an extension module the spec name
                                        # must match the PYBIND11_MODULE baked
                                        # into the .so, i.e. its stem
    spec = importlib.util.spec_from_file_location(bridge.stem, bridge)
    model.bridge = importlib.util.module_from_spec(spec)                      # type: ignore
    spec.loader.exec_module(model.bridge)                                     # type: ignore


def load_bdt(output_dir: Path, name: str) -> ModelBase:
    """Load a written BDT project and attach its compiled bridge."""
    model = load_model(output_dir / f"{name}.json")
    model.config.output_dir = str(output_dir)

    attach_bridge(model)

    return model


def load_grid(output_dir: str = "hls") -> NDArray:
    """Load a grid previously written and compiled by `convert_grid` +
    `compile`, skipping conversion and compilation. Shape
    is taken from the project directories.
    """
    root = Path(output_dir).resolve()

    n_steps  = len(list(root.glob("step*")))
    if not n_steps:
        raise FileNotFoundError(f"No step directories in {root}, convert a grid first")

    n_groups = len(list((root / "step00").glob("group*")))

    grid = [
        load_bdt(*project_paths(root, step, group))
        for step in range(n_steps)
        for group in range(n_groups)
    ]

    return np.array(grid, dtype = object).reshape(n_steps, n_groups)
