from __future__ import annotations

from os import cpu_count
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from conifer.model import load_model

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def compile_bdt(output_dir: Path | str, name: str) -> None:
    """Compile one already-written BDT project into its pybind11 bridge.

    Takes paths rather than the model so it can run in a worker process: conifer's
    `compile` chdirs, which not thread-safe. Bridge binding is process specific,
    so the parent rebinds it afterwards with `attach_bridge`.
    """
    output_dir = Path(output_dir)

    model = load_model(output_dir / f"{name}.json")
    model.config.output_dir = str(output_dir)

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
