from __future__ import annotations

from ..flowbdt import FlowBDT
from .utils import hls_config, project_paths

from pathlib import Path
import numpy as np
from conifer.converters import convert_from_xgboost
from conifer.model import ModelBase

from numpy.typing import NDArray
from xgboost import XGBModel

#-----------------------------------------------------------------------------

def convert_bdt(bdt: XGBModel, config: dict, output_dir: Path, name: str) -> ModelBase:
    """Convert one BDT into its own project, named `name`."""
    model = convert_from_xgboost(bdt, { **config,
                                        "OutputDir": str(output_dir),
                                        "ProjectName": name })
    
                                        # conifer stamps the pybind11 bridge with
                                        # the clock truncated to the second, 
                                        # which results in non-unique `.so` 
                                        # module names. 
                                        # Fix: manually overwrite with unique name.
    model._stamp = name                                                       # type: ignore

    return model        


def convert_grid(
    model: FlowBDT,
    config_overrides: dict | None = None,
    output_dir: str = "flowhls",
) -> NDArray:
    """Convert every BDT of `model.bdt_grid` to a conifer model, one project
    directory per BDT. Return an array of conifer models with the same
    (n_steps, n_groups) shape.
    """
    if model.multi_output:
        raise NotImplementedError(
            "Multi-output BDTs are not supported by conifer."
        )

    config = hls_config(**(config_overrides or {}))

    root = Path(output_dir).resolve()

    grid = [
        convert_bdt(bdt, config, *project_paths(root, step, group))
        for step, row in enumerate(model.bdt_grid)
        for group, bdt in enumerate(row)
    ]

    return np.array(grid, dtype = object).reshape(model.bdt_grid.shape)
