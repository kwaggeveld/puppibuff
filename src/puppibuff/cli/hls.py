from __future__ import annotations

from ..archive import load_model, save_samples
from ..utils import initial_noise
from .common import COUNT, EXISTING_FILE, N_SAMPLES_DEFAULT, OUTPUT_DIR, timed

from pathlib import Path

import click

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codecs import Codec
    from ..flowbdt import FlowBDT
    from ..hls import FlowHLS

#-----------------------------------------------------------------------------

def build_hls(model: FlowBDT, codec: Codec, workdir: str,
              merged: bool = True) -> FlowHLS:
    """Bind a design in `workdir` that can be sampled from, reusing compiled and
    written designs.
    """
    from ..hls import FlowHLS
    from ..hls.utils import is_compiled, is_written

    if is_compiled(workdir):
        return timed("Loading compiled grid", FlowHLS.load, workdir)

    if is_written(workdir):
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
@click.argument("model", type = EXISTING_FILE)
@click.option("-o", "--output", type = click.Path(file_okay = False),
              help = "HLS project directory  [default: ./output/hls/<model>/]")
@click.option("--per-bdt", is_flag = True,
              help = "Write one conifer project per BDT.")
def write(model: str, output: str | None, per_bdt: bool) -> None:
    """Write MODEL's HLS firmware to DIRECTORY."""
    from ..hls import FlowHLS

    _, codec, flowbdt = load_model(model)

    if output is None:                  # Beside the figures, one dir per model
        output = str(Path(OUTPUT_DIR) / "hls" / Path(model).stem)

    flowhls = FlowHLS.convert(flowbdt, output_dir = output, merged = not per_bdt)
    flowhls.write(codec)

    click.echo(f"Wrote { flowhls.output_dir }.")


@hls.command()
@click.argument("workdir", type = click.Path(exists = True, file_okay = False))
def build(workdir: str) -> None:
    """Synthesise the design already written into WORKDIR, then its VHDL payload.

    Run `puppibuff hls write` first. Requires `vitis_hls` on PATH.
    """
    from ..hls import constants, FlowHLS

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
@click.argument("model", type = EXISTING_FILE)
@click.option("-n", "--n-samples", type = COUNT, default = N_SAMPLES_DEFAULT,
              show_default = True,
              help = "Events to generate with each sampler from shared noise.")
@click.option("--encoded", is_flag = True,
              help = "Save the codec's input instead of its output")
def sample(workdir: str, model: str, n_samples: int, encoded: bool) -> None:
    """Sample MODEL through both HLS and XGBoost from shared noise.

    Saved into WORKDIR as decoded channels, one array per sampler per channel,
    with MODEL's config. Kept encoded if `--encoded`. `puppibuff plot` draws
    the decoded samples.
    """
    from ..hls import constants

    config, codec, flowbdt = load_model(model)

    flowhls = build_hls(flowbdt, codec, workdir)

    x0 = initial_noise((n_samples, flowhls.n_channels))

                                        # HLS first so ratios read HLS/target
    samples = {                         # and HLS/XGBoost
        "HLS": timed(f"Sampling { n_samples } with hls", flowhls.sample,
                     x0 = x0, solver = constants.SAMPLE_SOLVER),
        "XGBoost": timed(f"Sampling { n_samples } with XGBoost", flowbdt.sample,
                         x0 = x0, solver = constants.SAMPLE_SOLVER),
    }

    outdir = Path(workdir)
    path   = outdir / f"{ outdir.name }_samples.npz"

    save_samples(str(path), config, { label: raw if encoded else codec.decode(raw)
                                      for label, raw in samples.items() })

    click.echo(f"Wrote { path } ({ path.stat().st_size / 1e6 :.1f} MB).")
