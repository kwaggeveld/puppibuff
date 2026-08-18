from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # Runtime imports are deferred into
    from .codecs import Codec           # from_zip: flowbdt imports this module
    from .configs import Config
    from .flowbdt import FlowBDT

#-----------------------------------------------------------------------------

CONFIG_FILE = "config"                  # Members to_zip/from_zip agree on
CODEC_FILE  = "codec"
MODEL_FILE  = "flowbdt"

def fill_template(package: str, name: str, /, **fields) -> str:
    """Read `package`'s template `firmware/name` and substitute tokens `**field`
    Raise if given tokens are not equal to the expected tokens.
    """
    template = (resources.files(package) / "firmware" / name).read_text()

    token = re.compile(r"\*\*(\w+)\*\*")   # `**name**`
    found_fields = { match[1] for match in token.finditer(template) }
    given_fields = set(fields)

    if found_fields != given_fields:
        raise KeyError(
            f"Incorrect tokens for { package }/firmware/{ name }: "
            f"unfilled { found_fields - given_fields }, "
            f"unused { given_fields - found_fields }."
        )

    return token.sub(lambda match: str(fields[match[1]]), template)


def t_to_step(t: float, n_steps: int) -> int:
    """Snap `t` in [0, 1] to the nearest of `n_steps` integer time steps."""
    return int(np.floor(t * (n_steps - 1) + 0.5 + 1e-6))


def initial_noise(
        shape: tuple[int, int] | None,
        x0: NDArray | None = None,
        rng: np.random.Generator = np.random.default_rng()
    ) -> NDArray:
    """Return ND Gaussian noise drawn here if `x0` not given."""
    if x0 is not None:
        return x0

    if shape is None:
        raise ValueError("Provide either shape or initial noise x0.")

    return rng.standard_normal(shape, dtype = np.float32)


def to_zip(path: str, config: Config, codec: Codec, model: FlowBDT) -> None:
    """Export a trained run as one archive: config, codec and model."""
    Path(path).parent.mkdir(parents = True, exist_ok = True)

    with TemporaryDirectory() as tmp, ZipFile(path, "w", ZIP_DEFLATED) as archive:
        staged = Path(tmp)              # Write to a tempdir first

        config.to_json(str(staged / CONFIG_FILE))
        codec.to_json(str(staged / CODEC_FILE))
        model.to_disk(str(staged / MODEL_FILE))
                                        # Compress from tempdir into zip
        for name in (CONFIG_FILE, CODEC_FILE, MODEL_FILE):
            archive.write(staged / name, name)


def from_zip(path: str) -> tuple[Config, Codec, FlowBDT]:
    """Read back an archive written by `to_zip`."""
    from .codecs import Codec           # Deferred to prevent circular import
    from .configs import Config
    from .flowbdt import FlowBDT

    with TemporaryDirectory() as tmp, ZipFile(path) as archive:
        archive.extractall(tmp)         # Extract zip to tempdir first

        staged = Path(tmp)              # Read tempdir to load
        config = Config.from_json(str(staged / CONFIG_FILE))
        codec  = Codec.from_json(str(staged / CODEC_FILE))
        model  = FlowBDT.from_disk(str(staged / MODEL_FILE))

    return config, codec, model
