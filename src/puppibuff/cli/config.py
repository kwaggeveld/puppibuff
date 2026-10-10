from __future__ import annotations

from ..archive import load_config
from .common import config_names, resolve_config

import json
from pathlib import Path

import click

#-----------------------------------------------------------------------------

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
    
    try:
        config = load_config(source) if Path(source).is_file() else resolve_config(source)()
    except ValueError as error:
        raise click.BadParameter(str(error))

    click.echo(json.dumps(config.to_dict(), indent = 2))
