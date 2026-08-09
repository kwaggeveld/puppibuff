from __future__ import annotations

from .utils import bridge_path, project_paths

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import numpy as np
from conifer.model import load_model

from conifer.model import ModelBase
from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def import_bridge(bridge: Path):
    """Import an already-compiled pybind11 bridge by file location."""
    if not bridge.exists():
        raise FileNotFoundError(f"{bridge} not found, compile it first")

    spec   = spec_from_file_location(bridge.stem, bridge)
    module = module_from_spec(spec)                                           # type: ignore
    spec.loader.exec_module(module)                                           # type: ignore

    return module


def attach_bridge(model: ModelBase) -> None:
    """Attach an already-compiled pybind11 bridge to `model`.

    This is roughly the tail of conifer's `XilinxHLSModel.compile()`,
    which always rewrites and re-compiles.
    """
    config = model.config

    model.bridge = import_bridge(                                             # type: ignore
        bridge_path(config.output_dir, config.project_name)                   # type: ignore
    )


def load_bdt(output_dir: Path, name: str, attach: bool = True) -> ModelBase:
    """Load a written BDT project, and its compiled bridge if `attach`."""
    model = load_model(output_dir / f"{name}.json")
    model.config.output_dir = str(output_dir)

    if attach:
        attach_bridge(model)

    return model


def load_grid(output_dir: str | Path = "flowhls", attach: bool = True) -> NDArray:
    """Load a grid previously written and compiled by `convert_grid` +
    `compile`, skipping conversion and compilation. Shape is taken from
    the project directories.
    """
    root = Path(output_dir).resolve()

    n_steps  = len(list(root.glob("step*")))
    if not n_steps:
        raise FileNotFoundError(f"No step directories in {root}, convert a grid first.")

    n_groups = len(list((root / "step00").glob("group*")))

    grid = [
        load_bdt(*project_paths(root, step, group), attach = attach)
        for step in range(n_steps)
        for group in range(n_groups)
    ]

    return np.array(grid, dtype = object).reshape(n_steps, n_groups)
