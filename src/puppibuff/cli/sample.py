from __future__ import annotations

from ..archive import save_samples
from .common import (default_out, encoded_option, EXISTING_FILE, load_archive,
                     n_samples_option, OUTPUT_DIR, seed_option, timed)

import click
import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codecs import Codec
    from ..configs import Config

#-----------------------------------------------------------------------------

def sample_model(model: str, n_samples: int,
                 seed: int | None) -> tuple[Config, Codec, NDArray]:
    """Load `model` and integrate its field from noise."""
    config, codec, flowbdt = load_archive(model)

                                        # Overrides the config's seed
    rng = None if seed is None else np.random.default_rng(seed)

    raw = timed(f"Sampling { n_samples }", flowbdt.sample, n_samples, rng = rng)

    return config, codec, raw


@click.command()
@click.argument("model", type = EXISTING_FILE)
@n_samples_option
@seed_option
@encoded_option
@click.option("-o", "--output", type = click.Path(dir_okay = False),
              help = f"Output file  [default: ./{ OUTPUT_DIR }/samples/<model>.npz]")
def sample(model: str, n_samples: int, seed: int | None, encoded: bool,
           output: str | None) -> None:
    """Sample from MODEL into an `.npz` file that can be drawn with
    `puppibuff plot`.

    The output file holds one array per decoded channel and MODEL's config.
    `--encoded` keeps the codec's encoded input instead.
    """
    config, codec, raw = sample_model(model, n_samples, seed)

    if output is None:
        output = str(default_out("samples", model, ".npz"))

    path = save_samples(output, config,
                        { "Output": raw if encoded else codec.decode(raw) })

    click.echo(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB).")
