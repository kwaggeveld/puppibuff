from __future__ import annotations

from ..utils import FlowBDT

from pathlib import Path

import numpy as np

from conifer.backends.xilinxhls import auto_config
from conifer.converters import convert_from_xgboost
from conifer.model import ModelBase

from numpy.typing import NDArray
from xgboost import XGBModel

#-----------------------------------------------------------------------------

XILINX_PART = "xcvu13p-flga2577-2-e"


def hls_config(**overrides) -> dict:
    """Return default `xilinxhls` config with `XILINX_PART` set.

    `Precision` is left alone. conifer's `ap_fixed<18,8>` truncates, costing half
    an LSB on every tree accumulated -- a systematic bias, measured at -0.012
    over 20 trees. `ap_fixed<18,8,AP_RND_CONV,AP_SAT>` cuts that ~3x (to -0.004)
    but roughly triples the worst-case residual (1.02 -> 3.12), rounded
    thresholds flipping more split decisions. Bias traded for tail, so not a
    clear win; revisit per channel against the real distributions.
    """
    config = auto_config()
    config["XilinxPart"] = XILINX_PART
    config.update(overrides)

    return config


def convert_bdt(bdt: XGBModel, config: dict, output_dir: Path, name: str) -> ModelBase:
    """Convert one BDT into its own project, named `name`."""
    return convert_from_xgboost(bdt, { **config, 
                                       "OutputDir": str(output_dir),
                                       "ProjectName": name })


def convert_grid(
    model: FlowBDT,
    config: dict | None = None,
    output_dir: Path | str = "hls",
) -> NDArray:
    """Convert every BDT of `model.bdt_grid` to a conifer model, one project 
    directory per BDT. Return an array of conifer models with the same 
    (n_steps, n_groups) shape.
    """
    if model.multi_output:
        raise NotImplementedError(
            "conifer models have a single output, and XGBoost cannot even dump "
            "multi_strategy = \"multi_output_tree\" boosters"
        )

    config = hls_config() if config is None else config

    root = Path(output_dir).resolve()

    grid = [
        convert_bdt(
            bdt, config,
            root / f"step{step:02d}" / f"group{group:03d}",
            f"bdt_s{step:02d}_g{group:03d}",
        )
        for step, row in enumerate(model.bdt_grid)
        for group, bdt in enumerate(row)
    ]

    return np.array(grid, dtype = object).reshape(model.bdt_grid.shape)
