from ..utils import output_dir

import sys
import time
from pathlib import Path

import click

#-----------------------------------------------------------------------------

ERASE_LINE = "\r\x1b[K"                 # Back to column 0, then clear the rest

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
    """A number of events, written as `N` or as `1eN`."""

    name = "count"

    def convert(self, value, param, ctx) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            self.fail(f"{ value !r} is not a number of events.", param, ctx)

COUNT = Count()


def figure_path(kind: str, source: str, output: str | None) -> Path:
    """`output`, or `./output/<kind>/`, with `source`'s stem as the file name."""
    outdir = Path(output) if output is not None else output_dir(kind)
    outdir.mkdir(parents = True, exist_ok = True)

    return outdir / f"{ Path(source).stem }.pdf"
