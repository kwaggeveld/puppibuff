from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs import Config

#-----------------------------------------------------------------------------

def apply_style() -> None:
    """Style drawn figures using the package style. Imports matplotlib on call 
    to reduce time needed for '--help'.
    """
    import matplotlib.pyplot as plt

    plt.style.use("puppibuff.style")


ERASE_LINE = "\r\x1b[K"                 # Back to column 0, then clear the rest

OUTPUT_DIR = "output"                   # Under the cwd

CONFIG_DEFAULT = "FlatPuppiJetConfig"

def timed(label: str, call, *args, **kwargs):
    """Announce a step before running, and report the time it took."""
    in_terminal = sys.stderr.isatty()   # Only overwrite in live terminals
    eraser = ERASE_LINE if in_terminal else ""

    click.echo(f"{ label }...", err = True, nl = not in_terminal)
    start  = time.time()
    result = call(*args, **kwargs)

    click.echo(f"{ eraser }{ label }: { time.time() - start :.1f} s.", err = True)

    return result


class Count(click.ParamType):
    """A number of events, written as `N` or as `1eN`. `all` is parsed to `None`."""

    name = "int"

    def __init__(self, all: str | None = None) -> None:
        self.all = all

        if all is not None:
            self.name = f"int|{ all }"

    def convert(self, value, param, ctx) -> int | None:
        if value == self.all:
            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            self.fail(f"{ value !r} is not a number of events.", param, ctx)

COUNT = Count()


def figure_path(kind: str, source: str, output: str | None) -> Path:
    """`output`, or `./output/<kind>/`, with `source`'s stem as the file name."""
    outdir = Path(output) if output is not None else Path(OUTPUT_DIR) / kind
    outdir.mkdir(parents = True, exist_ok = True)

    return outdir / f"{ Path(source).stem }.pdf"


def config_names() -> list[str]:
    """Every built-in config."""
    from .. import configs

    return [ name for name in configs.__all__ if name != "Config" ]


def resolve_config(spec: str) -> type[Config]:
    """The config class `spec` names, as a built-in's name or as a
    `module:Class` path.
    """
    from .. import configs
    from ..utils import import_class

    if ":" not in spec:                 # Built-in config, add its tag
        if spec not in config_names():
            raise click.BadParameter(
                f"Unknown config { spec !r}. Built in: "
                f"{ ', '.join(config_names()) }."
            )

        spec = f"{ configs.__name__ }:{ spec }"
    elif "" not in sys.path:
        sys.path.insert(0, "")

    try:
        return import_class(spec)
    except (AttributeError, ImportError) as error:
        raise click.BadParameter(f"Could not import { spec }: { error }.")
