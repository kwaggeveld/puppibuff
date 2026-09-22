from .. import from_zip
from ..analyses import plot_contours, plot_distributions, plot_histograms
from .common import COUNT, figure_path, timed

import click
import numpy as np

from matplotlib.figure import Figure
from typing import Callable

#-----------------------------------------------------------------------------

N_SAMPLES_DEFAULT = 1_000_000
PLOTTERS: dict[str, Callable[..., Figure]] = {
    "histograms":    plot_histograms,
    "distributions": plot_distributions,
    "contours":      plot_contours,
}


def plot_options(command):
    """The arguments every plotter takes. Applied in reverse so `--help` lists
    them in the order written here.
    """
    options = [
        click.argument("model", type = click.Path(exists = True)),
        click.option("-n", "--n-samples", type = COUNT,
                     default = N_SAMPLES_DEFAULT, show_default = True,
                     help = "Number of samples."),
        click.option("-s", "--seed", type = int,
                     help = "Seed for the sampler."),
        click.option("--train-overlay/--no-train-overlay", default = True,
                     show_default = True,
                     help = "Include training dataset overlay."),
        click.option("-o", "--output", type = click.Path(file_okay = False),
                     help = f"Output directory  [default: ./output/{ command.__name__ }/]"),
    ]

    for option in reversed(options):
        command = option(command)

    return command


def draw(plotter: str, model: str, n_samples: int, seed: int | None,
         train_overlay: bool, output: str | None) -> None:
    """Sample `model`'s archive and draw it against the dataset it was trained on."""
    config, codec, flowbdt = timed("Loading model", from_zip, model)

                                        # Overrides saved rng
    rng = None if seed is None else np.random.default_rng(seed)

    data = config.dataset()             # Uses tqdm

    raw     = timed(f"Sampling { n_samples }", flowbdt.sample, n_samples, rng = rng)
    samples = codec.decode(raw)
                                        # Convert string plotter to function call
    figure = timed(f"Drawing { plotter }", PLOTTERS[plotter], data, samples,
                   n_events = config.n_events if train_overlay else None)

    path = figure_path(plotter, model, output)
    figure.savefig(path, format = "pdf")

    click.echo(f"Wrote { path }.")


@click.group()
def plot() -> None:
    """Draw a trained model against the dataset it was trained on."""


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
