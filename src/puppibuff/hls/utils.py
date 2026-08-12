from __future__ import annotations

from . import constants as c

from pathlib import Path
from xml.etree import ElementTree

from conifer.backends.xilinxhls import auto_config as xilinxhls_config

#-----------------------------------------------------------------------------

def hls_config(**overrides) -> dict:
    """Return default `xilinxhls` config, overridden with our part and clock."""
    return (xilinxhls_config()
            | { "XilinxPart": c.XILINX_PART, "ClockPeriod": c.CLOCK_PERIOD }
            | overrides)


def project_paths(root: Path, step: int, group: int) -> tuple[Path, str]:
    """The project directory and BDT name at (step, group)."""
    return (root / c.BDT_DATA / f"step{step:02d}" / f"group{group:02d}",
            f"bdt_s{step:02d}_g{group:02d}")


def bridge_path(output_dir: Path | str, name: str) -> Path:
    """Name of the pybind11 bridge that conifer's `compile()` builds for project
    `name`.
    """
    return Path(output_dir) / f"conifer_bridge_{name}.so"


def merged_bridge(root: Path | str) -> Path:
    """The merged design's bridge. Handles the constant `BRIDGE_MODULE` here."""
    return bridge_path(root, c.BRIDGE_MODULE)


def block_dir(root: Path | str, block: str) -> Path:
    """Construct a block's HLS project directory."""
    return Path(root) / c.BLOCKS_DIR / block


def block_latency(root: Path | str, block: str) -> int:
    """Read a block's latency in clock cycles from its synthesis report."""
    report = (block_dir(root, block) / block / "solution1" / "syn" / "report" 
              / "csynth.xml")

    if not report.exists():
        raise FileNotFoundError(
            f"{ block } has not been synthesised, { report } is missing. "
            f"Run ./build_all.sh in { root } first."
        )

    latency = ElementTree.parse(report).findtext(c.LATENCY_KEY)

    if latency is None:
        raise RuntimeError(f"No { c.LATENCY_KEY } in { report }.")

    return int(latency)


def merged_build(root: Path | str) -> bool:
    """Whether a merged design was compiled into `root`. The per-BDT layout keeps
    its bridges down in the project directories, so which `.so` sits at the root
    is what tells the two apart.
    """
    return merged_bridge(root).exists()
