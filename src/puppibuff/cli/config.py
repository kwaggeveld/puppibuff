from __future__ import annotations

from .common import config_names, resolve_config

import json
from pathlib import Path
from zipfile import is_zipfile, ZipFile

import click

#-----------------------------------------------------------------------------

def file_config(path: str) -> dict:
    """Resolve the config saved in a trained archive or a loose config JSON."""
    from ..configs import Config
    from ..utils import CONFIG_FILE

    if not is_zipfile(path):
        return Config.from_json(path).to_dict()

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
    """Print SOURCE's fields as JSON given a config name, a trained archive
    or a config JSON file.

    Pipe it into a file to edit and use it as `train --config-file`.
    """

    fields = (file_config(source) if Path(source).exists()
              else resolve_config(source)().to_dict())

    click.echo(json.dumps(fields, indent = 2))
