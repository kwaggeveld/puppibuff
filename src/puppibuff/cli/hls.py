from .. import from_zip
from ..analyses import plot_histograms
from ..codecs import Codec
from ..flowbdt import FlowBDT
from ..hls import constants, FlowHLS
from ..hls.utils import is_compiled
from ..utils import initial_noise, output_dir
from .common import COUNT, figure_path, timed

from pathlib import Path

import click
import numpy as np
from numpy.lib.npyio import NpzFile

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

N_SAMPLES_DEFAULT = 1_000_000           # Drawn once, run through both samplers

SAMPLERS = ( "hls", "xgb" )             # `sample` writes both, `plot` reads both


def sample_members(samples: dict[str, NDArray],
                   codec: Codec | None) -> dict[str, NDArray]:
    """Reshape and optionally decode the samples to be stored in a `.npz` file.

    Decoded, store one decoded array per channel per sampler named as
    `<sampler>_<channel>`. With no codec, each encoded array is stored under the
    sampler's name instead.
    """
    if codec is None:
        return { sampler: sample.astype(np.float32)
                 for sampler, sample in samples.items() }

    return { f"{ sampler }_{ channel }": values.astype(np.float32)
             for sampler, sample in samples.items()
             for channel, values in codec.decode(sample).items() }


def read_samples(arrays: NpzFile, codec: Codec) -> dict[str, dict[str, NDArray]]:
    """Both samplers' decoded channels out of a file `sample` wrote. 

    An `--encoded` file holds one array per sampler and is decoded now. A
    decoded one already has the channels so is left alone.
    """
    samples = {}

    for sampler in SAMPLERS:
        prefix = f"{ sampler }_"

        if sampler in arrays.files:     # Encoded, so decode it now
            samples[sampler] = codec.decode(arrays[sampler])
        else:                           # Filter the `sampler`'s outputs
            samples[sampler] = { name.removeprefix(prefix): arrays[name]
                                 for name in arrays.files
                                 if name.startswith(prefix) }

        if not samples[sampler]:
            raise click.BadParameter(
                f"No { sampler } sample in this file. Generate new samples with "
                f"`puppibuff hls sample`."
            )

    return samples


def build_hls(model: FlowBDT, codec: Codec, workdir: str,
              merged: bool = True) -> FlowHLS:
    """Bind a design in `workdir` that can be sampled from, reusing compiled and
    written designs.
    """
    if is_compiled(workdir):
        return timed("Loading compiled grid", FlowHLS.load, workdir)

    if Path(workdir).exists() and any(Path(workdir).rglob("bdt_s*_g*.json")):
        flowhls = timed("Loading written design", FlowHLS.load, workdir,
                        attach = False)
    else:
        flowhls = timed("Converting grid", FlowHLS.convert, model,
                        output_dir = workdir, merged = merged)
        timed("Writing", flowhls.write, codec)

    timed("Compiling", flowhls.compile)

    return flowhls


@click.group()
def hls() -> None:
    """Interface for HLS firmware writers and C-simulation."""


@hls.command()
@click.argument("model", type = click.Path(exists = True))
@click.option("-o", "--output", type = click.Path(file_okay = False),
              help = "HLS project directory  [default: ./output/hls/<model>/]")
@click.option("--per-bdt", is_flag = True,
              help = "Write one conifer project per BDT.")
def write(model: str, output: str | None, per_bdt: bool) -> None:
    """Write MODEL's HLS firmware to DIRECTORY."""
    _, codec, flowbdt = from_zip(model)

    if output is None:                  # Beside the figures, one dir per model
        output = str(output_dir("hls") / Path(model).stem)

    flowhls = FlowHLS.convert(flowbdt, output_dir = output, merged = not per_bdt)
    flowhls.write(codec)

    click.echo(f"Wrote { flowhls.output_dir }.")


@hls.command()
@click.argument("workdir", type = click.Path(exists = True, file_okay = False))
def build(workdir: str) -> None:
    """Synthesise the design already written into WORKDIR, then its VHDL payload.

    Run `puppibuff hls write` first. Requires `vitis_hls` on PATH.
    """
                                        # Nothing here samples, so the design
                                        # need not have been compiled
    flowhls = FlowHLS.load(workdir, attach = False)

    timed("Synthesising", flowhls.build)

    if not flowhls.merged:              # A per-BDT layout has no blocks to tie
        click.echo("Per-BDT layout: no payload to write.")
        return

    flowhls.write_payload()

    click.echo(f"Wrote { flowhls.output_dir / constants.PAYLOAD_FILE }.")


@hls.command()
@click.argument("workdir", type = click.Path(file_okay = False))
@click.argument("model", type = click.Path(exists = True))
@click.option("-n", "--n-samples", type = COUNT, default = N_SAMPLES_DEFAULT,
              show_default = True, help = "Events to generate with each sampler.")
@click.option("--encoded", is_flag = True,
              help = "Save the codec's input instead of its output")
def sample(workdir: str, model: str, n_samples: int, encoded: bool) -> None:
    """Sample MODEL through both HLS and XGBoost from shared noise.

    Saved into WORKDIR as decoded channels, one array per sampler per channel.
    Kept encoded if `--encoded`. `puppibuff hls plot` draws, and optinally 
    decodes, these generated samples.
    """
    _, codec, flowbdt = from_zip(model)

    flowhls = build_hls(flowbdt, codec, workdir)

    x0 = initial_noise((n_samples, flowhls.n_channels))

    samples = {
        "hls": timed(f"Sampling { n_samples } with hls", flowhls.sample,
                     x0 = x0, solver = constants.SAMPLE_SOLVER),
        "xgb": timed(f"Sampling { n_samples } with XGBoost", flowbdt.sample,
                     x0 = x0, solver = constants.SAMPLE_SOLVER),
    }

    path    = Path(workdir) / f"{ Path(workdir).name }_samples.npz"
    members = sample_members(samples, None if encoded else codec)
    
    np.savez(path, **members)                                                   # type: ignore[arg-type]

    click.echo(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB).")


@hls.command()
@click.argument("samples", type = click.Path(exists = True, dir_okay = False))
@click.argument("model", type = click.Path(exists = True))
@click.option("-o", "--output", type = click.Path(file_okay = False),
              help = "Output directory  [default: ./output/hls/]")
def plot(samples: str, model: str, output: str | None) -> None:
    """Draw HLS/XGBoost's SAMPLES against the dataset MODEL was trained on."""
    arrays = np.load(samples)

    config, codec, _ = timed("Loading model", from_zip, model)

    sampled = read_samples(arrays, codec)

    data = config.dataset()             # Uses tqdm

                                        # HLS takes the primary slot so ratios
                                        # read HLS/target and HLS/XGBoost
    figure = timed("Drawing histograms", plot_histograms, data,
                   sample  = sampled["hls"],
                   overlay = sampled["xgb"],
                   labels  = { "Output": "HLS", "Training": "Python" })

    path = figure_path("hls", samples, output)
    figure.savefig(path, format = "pdf")

    click.echo(f"Wrote { path }.")
