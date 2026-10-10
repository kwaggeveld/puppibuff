from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import is_zipfile, ZipFile, ZIP_DEFLATED

from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # Runtime imports for CLI
    from .codecs import Codec
    from .configs import Config
    from .flowbdt import FlowBDT

#-----------------------------------------------------------------------------

CONFIG_FILE = "config"
CODEC_FILE  = "codec"
MODEL_FILE  = "flowbdt"


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
