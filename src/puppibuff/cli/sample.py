from __future__ import annotations

from ..archive import is_archive, load_model, save_samples
from .common import COUNT, EXISTING_FILE, N_SAMPLES_DEFAULT, OUTPUT_DIR, timed

from pathlib import Path

import click
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codecs import Codec
    from ..configs import Config

#-----------------------------------------------------------------------------

def sample_path(model: str, output: str | None) -> Path:
    """Construct the path to the output file. `output`, if provided, else 
    `./output/samples/<model>.npz`."""
    path = (Path(output) if output is not None
            else Path(OUTPUT_DIR) / "samples" / f"{ Path(model).stem }.npz")

    path.parent.mkdir(parents = True, exist_ok = True)

    return path


def sample_model(model: str, n_samples: int,
                 seed: int | None) -> tuple[Config, Codec, NDArray]:
    """Load `model` and integrate its field from noise."""
    if not is_archive(model):
        raise click.BadParameter(
            f"{ model } is no model archive. Write one with `puppibuff train`."
        )

    config, codec, flowbdt = timed("Loading model", load_model, model)

                                        # Overrides the config's seed
    rng = None if seed is None else np.random.default_rng(seed)

    raw = timed(f"Sampling { n_samples }", flowbdt.sample, n_samples, rng = rng)

    return config, codec, raw


@click.command()
@click.argument("model", type = EXISTING_FILE)
@click.option("-n", "--n-samples", type = COUNT,
              default = N_SAMPLES_DEFAULT, show_default = True,
              help = "Number of samples.")
@click.option("-s", "--seed", type = int, help = "Set sampler seed.")
@click.option("--encoded", is_flag = True,
              help = "Save the Codec's input instead of its output.")
@click.option("-o", "--output", type = click.Path(dir_okay = False),
              help = "Output file  [default: ./output/samples/<model>.npz]")
def sample(model: str, n_samples: int, seed: int | None, encoded: bool,
           output: str | None) -> None:
    """Sample from MODEL into an `.npz` file that can be drawn with
    `puppibuff plot`.

    The output file holds one array per decoded channel and MODEL's config.
    `--encoded` keeps the codec's encoded input instead.
    """
    config, codec, raw = sample_model(model, n_samples, seed)

    path = save_samples(str(sample_path(model, output)), config,
                        { "Output": raw if encoded else codec.decode(raw) })

    click.echo(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB).")
