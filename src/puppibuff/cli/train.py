from .. import to_zip

import click

#-----------------------------------------------------------------------------

@click.command()
@click.argument("outfile", type = click.Path(dir_okay = False))
def train(outfile: str) -> None:
    """Train a FlowBDT model and export it to OUTFILE.

    One archive holds the config, codec and model together, read back by
    `puppibuff.from_zip`.
    """
    from ..configs import FlatPuppiJetConfig

    config = FlatPuppiJetConfig()

    _, codec, model, x, y = config.setup()

    model.fit(x, y)

    to_zip(outfile, config, codec, model)

    click.echo(f"Wrote config, codec and model to { outfile }.")
