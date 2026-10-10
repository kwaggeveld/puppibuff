from __future__ import annotations

from ..archive import is_archive, load_samples
from .common import (apply_style, default_out, EXISTING_FILE, given,
                     n_samples_option, OUTPUT_DIR, seed_option, timed)
from .sample import sample_model

import click
from pathlib import Path
from zipfile import is_zipfile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils import Source

#-----------------------------------------------------------------------------

@click.command()
@click.argument("kind", type = click.Choice(( "histograms", "distributions", "contours" )),
                metavar = "KIND")
@click.argument("source", type = EXISTING_FILE)
@n_samples_option
@seed_option
@click.option("--train-overlay", type = bool,
              default = True, show_default = True,
              help = "Include training dataset overlay, unless SOURCE holds two samples.")
@click.option("-o", "--output", type = click.Path(file_okay = False),
              help = f"Output directory  [default: ./{ OUTPUT_DIR }/KIND/]")
@click.option("--show", is_flag = True,
              help = "Show the figure instead of writing, ignoring -o.")
def plot(kind: str, source: str, n_samples: int, seed: int | None,
         train_overlay: bool, output: str | None, show: bool) -> None:
    """Draw a model or samples by `puppibuff sample` or `puppibuff hls sample`
    against its target dataset, as binned `histograms`, KDE `distributions` or
    2D KDE `contours`.

    SOURCE is either a model archive, which is sampled first with `-n` and
    `-s`, or an `.npz` of premade samples.
    """
    from .. import analyses

    apply_style()

    if not is_zipfile(source):
        raise click.BadParameter(
            f"{ source } is neither a model archive nor an `.npz` of samples."
        )

    series: dict[str, Source]

    if is_archive(source):
        config, codec, raw = sample_model(source, n_samples, seed)
        series = { "Output": codec.decode(raw) }
    else:
        if given("n_samples", "seed"):
            raise click.BadParameter(
                f"Pass `-n` and `-s` only with a model archive. "
                f"`{ source }` already holds samples."
            )

        try:
            config, samples = load_samples(source)
        except ValueError as error:
            raise click.BadParameter(str(error))

        if not all(isinstance(sample, dict) for sample in samples.values()):
            raise click.BadParameter(
                "These samples are encoded. Sample again without `--encoded` to plot."
            )

        series = { label: sample for label, sample in samples.items() if isinstance(sample, dict) }

    data = config.dataset()             # Uses tqdm

    if train_overlay and len(series) == 1:
        series["Training"] = data[:config.n_events]

    figure = timed(f"Drawing { kind }", getattr(analyses, f"plot_{ kind }"), data, series)

    if show:
        import matplotlib.pyplot as plt

        plt.show()
        return

    path = default_out(kind, source, ".pdf")
    if output is not None:
        path = Path(output) / path.name

    path.parent.mkdir(parents = True, exist_ok = True)
    figure.savefig(path, format = "pdf")

    click.echo(f"Wrote { path }.")
