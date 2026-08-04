from __future__ import annotations

from pathlib import Path

from conifer.backends.xilinxhls import auto_config as xilinxhls_config

#-----------------------------------------------------------------------------

XILINX_PART = "xcvu13p-flga2577-2-e"
FLOWHLS_PROJECT = "flowhls"             # Names the merged design's bridge and the
                                        # tcl `prj_name`.

def hls_config(**overrides) -> dict:
    """Return default `xilinxhls` config with `XILINX_PART` set."""
    return xilinxhls_config() | { "XilinxPart": XILINX_PART } | overrides


def project_paths(root: Path, step: int, group: int) -> tuple[Path, str]:
    """The project directory and BDT name at (step, group)."""
    return (root / f"step{step:02d}" / f"group{group:03d}",
            f"bdt_s{step:02d}_g{group:03d}")


def project_root(project_dir: Path | str) -> Path:
    """`project_paths`' inverse: the root a BDT project sits under. Absolute,
    since conifer is always given an absolute `OutputDir`.
    """
    return Path(project_dir).parents[1]


def bridge_path(output_dir: Path | str, name: str) -> Path:
    """Name of the pybind11 bridge that conifer's `compile()` builds for project
    `name`.
    """
    return Path(output_dir) / f"conifer_bridge_{name}.so"


def merged_build(root: Path | str) -> bool:
    """Whether a merged design was compiled into `root`. The per-BDT layout keeps
    its bridges down in the project directories, so which `.so` sits at the root
    is what tells the two apart.
    """
    return bridge_path(root, FLOWHLS_PROJECT).exists()
