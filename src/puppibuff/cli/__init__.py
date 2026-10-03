from .config import config
from .hls import hls
from .plot import plot
from .train import train

import click

#-----------------------------------------------------------------------------

CONTEXT_SETTINGS = { "max_content_width": 120 }

@click.group(context_settings = CONTEXT_SETTINGS)
@click.version_option(package_name = "puppibuff")
def main() -> None:
    """BDT-based flow matching for on-the-fly event generation on FPGA firmware.

    Train a model, plot it against its dataset, and translate it to HLS.
    """

main.add_command(train)
main.add_command(config)
main.add_command(plot)
main.add_command(hls)

__all__ = [
    "main",
]
