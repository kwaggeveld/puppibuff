from __future__ import annotations

from ..archive import CONFIG_FILE, is_archive
from .common import config_names, resolve_config
from .sample import config_json

import json
from pathlib import Path
from zipfile import is_zipfile, ZipFile

import click
import numpy as np

#-----------------------------------------------------------------------------

def file_config(path: str) -> dict:
    """Resolve the config saved in a trained archive, a sample file, or a loose
    config JSON.
    """
    from ..configs import Config

    if not is_zipfile(path):            # Loose config JSON
        try:
            return Config.load(path).to_dict()
        except json.JSONDecodeError:
            raise click.BadParameter(f"{ path } is no config JSON.")

    if not is_archive(path):            # Samples
        return config_json(np.load(path))

    try:
        with ZipFile(path) as archive:
            return json.loads(archive.read(CONFIG_FILE))
    except KeyError:
        raise click.BadParameter(f"{ path } holds no { CONFIG_FILE } file, "
                                 f"write it with `train`.")


@click.group()
def config() -> None:
    """Inspect a training configuration."""


@config.command(name = "list")
def list_configs() -> None:
    """List every config built into puppibuff."""
    for name in config_names():
        click.echo(name)


@config.command()
@click.argument("source")
def show(source: str) -> None:
    """Print SOURCE's config fields as JSON given a config name, a trained archive,
    a file by `puppibuff sample` or a config JSON file.

    Pipe it into a file to edit and use it as `train --config-file`.
    """
    
    fields = (file_config(source) if Path(source).is_file()
              else resolve_config(source)().to_dict())

    click.echo(json.dumps(fields, indent = 2))
