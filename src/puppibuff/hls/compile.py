from __future__ import annotations

from .utils import BRIDGE_MODULE, bridge_path

from os import cpu_count, environ, system
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from conifer.model import load_model
from conifer.utils import _ap_include, _gcc_opts, _py_executable

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def compile_bdt(output_dir: str, name: str) -> None:
    """Compile one already-written BDT project into its pybind11 bridge.

    Takes paths rather than the model so it can run in a worker process: conifer's
    `compile` chdirs, which not thread-safe. Bridge binding is process specific,
    so the parent rebinds it afterwards with `attach_bridge`.
    """
    model = load_model(Path(output_dir) / f"{ name }.json")
    model.config.output_dir = output_dir

    model._stamp = name                 # See `convert_bdt`                   # type: ignore

    model.compile()


def compile_grid(grid: NDArray, n_threads: int | None = None) -> None:
    """Compile every BDT of `grid`, `n_threads` at a time."""
    n_threads = n_threads or cpu_count()

    jobs = (
        delayed(compile_bdt)(model.config.output_dir, model.config.project_name)
        for model in grid.flat
    )

    with tqdm(total = grid.size, desc = "compile") as progress_bar:
        for _ in Parallel(n_jobs = n_threads, return_as = "generator_unordered")(jobs):
            progress_bar.update()


_flowhls_build: Path | None = None

def compile_flowhls(output_dir: Path, files: list[str]) -> None:
    """Compile the merged design's sources into its pybind11 bridge.

    `output_dir` is `FlowHLS.output_dir`, already absolute; `files` are its
    translation units, relative to it.
    """
    global _flowhls_build

    if _flowhls_build is not None:
        raise RuntimeError(
            f"A project was already compiled in this process "
            f"({_flowhls_build}). CPython caches the bridge by module name, so "
            f"binding a second build would silently return the first one's. "
            f"Run each build in its own process."
        )

    ap_include = _ap_include()
    if ap_include is None:
        raise RuntimeError(
            "Couldn't find Xilinx ap_ headers. Source the Vitis toolchain, "
            "or set XILINX_AP_INCLUDE."
        )

    CXX = environ.get("CXX", "g++")

    command = (f"cd { output_dir } && "
               f"{CXX} -O3 -shared -std=c++14 -fPIC "       # Matching conifer
               f"$({ _py_executable() } -m pybind11 --includes) { ap_include } { _gcc_opts() } "
               f"bridge.cpp { ' '.join(files) } -o { bridge_path('.', BRIDGE_MODULE) }")

    if system(command) != 0:
        raise RuntimeError(f"Failed to compile merged project in { output_dir }")
    
    print("Compiled merged FlowHLS.")
    
    _flowhls_build = output_dir
