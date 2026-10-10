from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import is_zipfile, ZipFile, ZIP_DEFLATED

import numpy as np

from numpy.typing import NDArray
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:                       # Runtime imports for CLI
    from numpy.lib.npyio import NpzFile
    from .codecs import Codec
    from .configs import Config
    from .flowbdt import FlowBDT

#-----------------------------------------------------------------------------

CONFIG_FILE = "config"
CODEC_FILE  = "codec"
MODEL_FILE  = "flowbdt"

                                        # Label -> decoded channels, or to
Samples: TypeAlias = "dict[str, dict[str, NDArray] | NDArray]"  # one encoded array


def is_archive(path: str) -> bool:
    """Distinguish between a model archive and file with samples."""
    if not is_zipfile(path):
        return False

    with ZipFile(path) as archive:      # `.npz` is a zip too
        return MODEL_FILE in archive.namelist()


def save_model(path: str, config: Config, codec: Codec, flowbdt: FlowBDT) -> None:
    """Export a trained run as one archive with config, codec and model."""
    Path(path).parent.mkdir(parents = True, exist_ok = True)

    with TemporaryDirectory() as tmp, ZipFile(path, "w", ZIP_DEFLATED) as archive:
        staged = Path(tmp)              # Write to a tempdir first

        config.save(str(staged / CONFIG_FILE))
        codec.save(str(staged / CODEC_FILE))
        flowbdt.save(str(staged / MODEL_FILE))
                                        # Compress from tempdir into zip
        for name in (CONFIG_FILE, CODEC_FILE, MODEL_FILE):
            archive.write(staged / name, name)


def load_model(path: str) -> tuple[Config, Codec, FlowBDT]:
    """Read back an archive written by `save_model`."""
    from .codecs import Codec
    from .configs import Config
    from .flowbdt import FlowBDT

    if not is_archive(path):
        raise ValueError(f"{ path } is not a model archive.")

    with TemporaryDirectory() as tmp, ZipFile(path) as archive:
        archive.extractall(tmp)         # Extract zip to tempdir first

        staged = Path(tmp)              # Read tempdir to load
        config  = Config.load(str(staged / CONFIG_FILE))
        codec   = Codec.load(str(staged / CODEC_FILE))
        flowbdt = FlowBDT.load(str(staged / MODEL_FILE))

    _, flowbdt.rng = config.rngs()

    return config, codec, flowbdt


def save_samples(path: str, config: Config, samples: Samples) -> Path:
    """Write labelled samples and the corresponding `config` to `path`.
    Return the path written.
    """
    members = { CONFIG_FILE: np.array(json.dumps(config.to_dict())) }

    for label, sample in samples.items():
        if '.' in label:
            raise ValueError(f"Sample label { label !r} contains '.'")

        if isinstance(sample, dict):    # Decoded: store as dict
            members |= { f"{ label }.{ channel }": values.astype(np.float32)
                         for channel, values in sample.items() }
        else:                           # Encoded: store as array
            members[label] = sample.astype(np.float32)

    written = Path(path if path.endswith(".npz") else f"{ path }.npz")
    written.parent.mkdir(parents = True, exist_ok = True)

    np.savez(written, **members)                                                # type: ignore[arg-type]

    return written


def _config(arrays: NpzFile, path: str) -> Config:
    """Load a config described by the sample."""
    from .configs import Config

    if CONFIG_FILE not in arrays.files:
        raise ValueError(f"{ path } holds no config.")

    return Config.from_dict(json.loads(arrays[CONFIG_FILE].item()))


def load_samples(path: str) -> tuple[Config, Samples]:
    """Read back the config and labelled samples from `save_samples`."""
    decoded: dict[str, dict[str, NDArray]] = {}
    encoded: dict[str, NDArray] = {}

    with np.load(path) as samples_file:
        for name in samples_file.files:
            if name == CONFIG_FILE:
                continue

            label, _, channel = name.partition('.')

            if channel:
                decoded.setdefault(label, {})[channel] = samples_file[name]
            else:
                encoded[label] = samples_file[name]

        return _config(samples_file, path), { **decoded, **encoded }


def load_config(path: str) -> Config:
    """The config of a model archive, a sample file or a loose config JSON."""
    from .configs import Config

    if is_archive(path):
        with ZipFile(path) as archive:
            return Config.from_dict(json.loads(archive.read(CONFIG_FILE)))

    if is_zipfile(path):
        with np.load(path) as samples_file:
            return _config(samples_file, path)

    try:
        return Config.load(path)
    except json.JSONDecodeError:
        raise ValueError(f"{ path } is not a model archive, sample file, or config JSON.") from None
