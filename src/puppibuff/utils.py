from __future__ import annotations

import json
import re
import warnings
from importlib import import_module, resources
from operator import attrgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Mapping, TypeAlias

if TYPE_CHECKING:                       # Runtime imports are done in `from_zip`.
    from .codecs import Codec           # `flowbdt` imports this file -> circular
    from .configs import Config
    from .datasets import Dataset
    from .flowbdt import FlowBDT

#-----------------------------------------------------------------------------

CONFIG_FILE = "config"                  # Members to_zip/from_zip agree on
CODEC_FILE  = "codec"
MODEL_FILE  = "flowbdt"

def output_dir(file: str) -> Path:
    """`file`'s directory / "output" / `file`'s stem, created if absent.

    Scripts call this as `output_dir(__file__)`.
    """
    path = Path(file).resolve().parent / "output" / Path(file).stem
    path.mkdir(parents = True, exist_ok = True)

    return path


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
        rng: np.random.Generator | None = None,
    ) -> NDArray:
    """Return ND Gaussian noise drawn here if `x0` not given."""
    if x0 is not None:
        return x0

    if shape is None:
        raise ValueError("Provide either shape or initial noise x0.")

    rng = np.random.default_rng() if rng is None else rng

    return rng.standard_normal(shape, dtype = np.float32)


def class_path(cls: type) -> str:
    """`cls` as an importable `module:QualName` tag for a JSON export."""
    if cls.__module__ == "__main__":
        warnings.warn(f"{ cls.__name__ } is defined in __main__, so it loads "
                      f"back only in a script that defines it.")

    return f"{ cls.__module__ }:{ cls.__qualname__ }"


def import_class(path: str) -> type:
    """Resolve a `module:QualName` tag written by `class_path`."""
    module, _, name = path.partition(":")

    return attrgetter(name)(import_module(module))


def _plain(value: object) -> object:
    """`json.dumps` fallback, converting numpy objects to lists and numbers."""
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()

    raise TypeError(f"{ type(value).__name__ } is not JSON serialisable")


def to_state(obj: object) -> dict:
    """Return `obj`'s class tag and every attribute as plain JSON."""
    state = { "cls": class_path(type(obj)) }

    for name, value in vars(obj).items():
        try:
            state[name] = json.loads(json.dumps(value, default = _plain))
        except TypeError as error:
            raise TypeError(f"Cannot save { type(obj).__name__ }.{ name }: { error }.") from None

    return state


def from_state(state: dict) -> object:
    """Rebuild the object from the dict constructed by `to_state`."""
    state = dict(state)
    obj: object = object.__new__(import_class(state.pop("cls")))
    obj.__dict__.update(state)

    return obj


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

    _, model.rng = config.rngs()

    return config, codec, model


Source: TypeAlias = "Dataset | Mapping[str, NDArray]"


def real_mask(mask: NDArray) -> NDArray:
    """Boolean form of a `real` channel."""
    return mask > .5


def multiplicity(mask: NDArray) -> NDArray:
    """Number of genuine constituents per jet from an `(N, M)` `real` channel."""
    return real_mask(mask).sum(axis = 1)


def flatten(source: Source, channels: list[str] | None = None) -> dict[str, NDArray]:
    """One 1-D array per channel. Padded jets with a `real` channel drop their
    padding and get a leading per-jet `multiplicity`.
    """
    channels = channels or [ channel for channel in source if channel != "real" ]

    if "real" not in source:
        return { channel: source[channel] for channel in channels }

    genuine = real_mask(source["real"])

    return ({ "multiplicity": multiplicity(source["real"]) }
            | { channel: source[channel][genuine] for channel in channels })
