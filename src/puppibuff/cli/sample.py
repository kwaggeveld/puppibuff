from __future__ import annotations

from .. import from_zip
from ..utils import CONFIG_FILE
from .common import (COUNT, EXISTING_FILE, is_archive, N_SAMPLES_DEFAULT,
                     OUTPUT_DIR, timed)

import json
from pathlib import Path

import click
import numpy as np
from numpy.lib.npyio import NpzFile

from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codecs import Codec
    from ..configs import Config

#-----------------------------------------------------------------------------

ENCODED_MEMBER = "encoded"              # Pre-decode output, a single array


def sample_members(config: Config,
                   samples: dict[str, NDArray]) -> dict[str, NDArray]:
    """Build the to-be-saved dict of one output array per channel and config dict."""
    channels = { channel: values.astype(np.float32)
                 for channel, values in samples.items() }

    return { CONFIG_FILE: np.array(json.dumps(config.to_dict())) } | channels


def config_json(zip: NpzFile) -> dict:
    """Return the config saved in `zip`."""
    if CONFIG_FILE not in zip.files:
        raise click.BadParameter(
            f"This file holds no { CONFIG_FILE } member, so the source model "
             "and target dataset are lost. Generate new samples with "
             "`puppibuff sample`."
        )

    return json.loads(zip[CONFIG_FILE].item())


def read_config(zip: NpzFile) -> Config:
    """Reconstruct and return the Config saved in `zip`."""
    from ..configs import Config

    return Config.from_dict(config_json(zip))


def read_samples(zip: NpzFile) -> dict[str, NDArray]:
    """Check whether samples are encoded, and if not, return all but the config."""
    if ENCODED_MEMBER in zip.files:
        raise click.BadParameter(
            "These samples are encoded and the required Codec was not stored. "
            "Draw again without `--encoded` to plot."
        )

    return { name: zip[name] for name in zip.files
             if name != CONFIG_FILE }


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
    if not is_archive(model):           # `from_zip` would fail on a temp path
        raise click.BadParameter(
            f"{ model } is no model archive. Write one with `puppibuff train`."
        )

    config, codec, flowbdt = timed("Loading model", from_zip, model)

                                        # Overrides the saved rng
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

    samples = { ENCODED_MEMBER: raw } if encoded else codec.decode(raw)
    path    = sample_path(model, output)

    np.savez(path, **sample_members(config, samples))                           # type: ignore[arg-type]

    click.echo(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB).")
