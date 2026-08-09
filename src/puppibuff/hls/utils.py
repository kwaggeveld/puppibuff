from __future__ import annotations

from pathlib import Path

from conifer.backends.xilinxhls import auto_config as xilinxhls_config

#-----------------------------------------------------------------------------

XILINX_PART = "xcvu13p-flga2577-2-e"
BRIDGE_MODULE = "flowhls"               # The merged design's pybind11 bridge
BDT_DATA = "bdt_data"                   # And the folder holding its BDT `.json`s

def hls_config(**overrides) -> dict:
    """Return default `xilinxhls` config with `XILINX_PART` set."""
    return xilinxhls_config() | { "XilinxPart": XILINX_PART } | overrides


def project_paths(root: Path, step: int, group: int) -> tuple[Path, str]:
    """The project directory and BDT name at (step, group)."""
    return (root / BDT_DATA / f"step{step:02d}" / f"group{group:02d}",
            f"bdt_s{step:02d}_g{group:02d}")


def bridge_path(output_dir: Path | str, name: str) -> Path:
    """Name of the pybind11 bridge that conifer's `compile()` builds for project
    `name`.
    """
    return Path(output_dir) / f"conifer_bridge_{name}.so"


def merged_bridge(root: Path | str) -> Path:
    """The merged design's bridge. Handles the constant `BRIDGE_MODULE` here."""
    return bridge_path(root, BRIDGE_MODULE)


def merged_build(root: Path | str) -> bool:
    """Whether a merged design was compiled into `root`. The per-BDT layout keeps
    its bridges down in the project directories, so which `.so` sits at the root
    is what tells the two apart.
    """
    return merged_bridge(root).exists()
