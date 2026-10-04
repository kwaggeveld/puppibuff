from .common import (apply_style, COUNT, EXISTING_FILE, figure_path,
                     is_archive, N_SAMPLES_DEFAULT, timed)
from .sample import read_config, read_samples, sample_model

from zipfile import is_zipfile

import click
import numpy as np

#-----------------------------------------------------------------------------

def plot_options(command):
    """The arguments every plotter takes. Applied in reverse so `--help` lists
    them in the order written here.
    """
    options = [
        click.argument("source", type = EXISTING_FILE),
        click.option("-n", "--n-samples", type = COUNT,
                     default = N_SAMPLES_DEFAULT, show_default = True,
                     help = "Number of samples, if SOURCE is a model."),
        click.option("-s", "--seed", type = int,
                     help = "Sampler seed, if SOURCE is a model."),
        click.option("--train-overlay", type = bool, 
                     default = True, show_default = True,
                     help = "Include training dataset overlay."),
        click.option("-o", "--output", type = click.Path(file_okay = False),
                     help = f"Output directory  [default: ./output/{ command.__name__ }/]"),
        click.option("--show", is_flag = True,
                     help = "Show the figure instead of writing, ignoring -o."),
    ]

    for option in reversed(options):
        command = option(command)

    return command


def refuse_options(*names: str) -> None:
    """Refuse options `*names` if they are set."""
    context = click.get_current_context()

    given = [ f"--{ name.replace('_', '-') }" for name in names
              if context.get_parameter_source(name)
                 is not click.ParameterSource.DEFAULT ]

    if given:
        raise click.BadParameter(
            f"Pass `{ '`, `'.join(given) }` only with a model archive. "
            f"`{ context.params['source'] }` already holds samples."
        )


def draw(
    plotter: str, 
    source: str, 
    n_samples: int, 
    seed: int | None,
    train_overlay: bool, 
    output: str | None,
    show: bool
) -> None:
    """Draw `source`'s samples against the target dataset behind them, sampling them
    first when `source` is a model rather than a sample file.
    """
    from ..analyses import plot_contours, plot_distributions, plot_histograms

    apply_style()

    plotters = {                        # The plotter each command uses
        "histograms":    plot_histograms,
        "distributions": plot_distributions,
        "contours":      plot_contours,
    }

    if not is_zipfile(source):
        raise click.BadParameter(
            f"{ source } is neither a model archive nor an `.npz` of samples."
        )

    if is_archive(source):
        config, codec, raw = sample_model(source, n_samples, seed)
        samples = codec.decode(raw)
    else:
        refuse_options("n_samples", "seed")

        arrays  = np.load(source)
        config  = read_config(arrays)
        samples = read_samples(arrays)

    data = config.dataset()             # Uses tqdm

    figure = timed(f"Drawing { plotter }", plotters[plotter], data, samples,
                   n_events = config.n_events if train_overlay else None)

    if show:
        import matplotlib.pyplot as plt

        plt.show()
        return

    path = figure_path(plotter, source, output)
    figure.savefig(path, format = "pdf")

    click.echo(f"Wrote { path }.")


@click.group()
def plot() -> None:
    """Draw a model or samples by `puppibuff sample` against its dataset.

    SOURCE is either a model archive, which is sampled first, or an `.npz` of
    samples, which get drawn immediately.
    """


@plot.command()
@plot_options
def histograms(**kwargs) -> None:
    """Binned marginal distributions with ratio panels."""
    draw("histograms", **kwargs)


@plot.command()
@plot_options
def distributions(**kwargs) -> None:
    """1D KDE marginal distributions."""
    draw("distributions", **kwargs)


@plot.command()
@plot_options
def contours(**kwargs) -> None:
    """Contours of 2D KDE marginal distributions."""
    draw("contours", **kwargs)
