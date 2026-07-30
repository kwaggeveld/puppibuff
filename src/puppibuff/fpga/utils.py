from __future__ import annotations

from pathlib import Path

from conifer.backends.xilinxhls import auto_config as xilinxhls_config

#-----------------------------------------------------------------------------

XILINX_PART = "xcvu13p-flga2577-2-e"


def hls_config(**overrides) -> dict:
    """Return default `xilinxhls` config with `XILINX_PART` set."""
    config = xilinxhls_config()
    config["XilinxPart"] = XILINX_PART
    config.update(overrides)

    return config


def project_paths(root: Path, step: int, group: int) -> tuple[Path, str]:
    """Construct project directory and BDT name at (step, group)."""
    return (root / f"step{step:02d}" / f"group{group:03d}",
            f"bdt_s{step:02d}_g{group:03d}")
