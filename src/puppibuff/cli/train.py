from __future__ import annotations

from .. import to_zip
from .common import CONFIG_DEFAULT, Count, resolve_config

import ast
import click

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs import Config

#-----------------------------------------------------------------------------

ALL_EVENTS = "all"                      # `Config` uses the whole dataset
                                        # with `n_events = None`

EVENTS = Count(ALL_EVENTS)


PASSTHROUGH_HELP = ( "--XGBKEY=VALUE", "XGBoost parameters, e.g. "
                                    "`--max-depth=8`." )

class Passthrough(click.Command):
    """Adds an option `--key=value` among the declared ones for documentation."""
    def format_options(self, ctx, formatter) -> None:
        records = [ record for param in self.get_params(ctx)
                    if (record := param.get_help_record(ctx)) is not None ]
        records.insert(-1, PASSTHROUGH_HELP)

        with formatter.section("Options"):
            formatter.write_dl(records)


def xgboost_names() -> set[str]:
    """Every name a bare `--key=value` accepts, from XGBoost's signature."""
    from xgboost import XGBRegressor

    return set(XGBRegressor().get_params())


def option_pair(option: str) -> tuple[str, object]:
    """Parse an `--key=value` string as an XGBoost parameter, convert kebab case
    to snake case.
    """
    key, separator, value = option.partition("=")

    if not option.startswith("-") or not separator:
        raise click.BadParameter(
            f"{ option !r} is neither an option of this command nor an "
            f"XGBoost parameter written as `--key=value`."
        )

    name = key.lstrip("-").replace("-", "_")

    try:                                # `device=cuda` is no literal
        return name, ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return name, value


def tree_params(passthrough: tuple[str, ...]) -> dict:
    """Convert unknown `--key=value` parameters to XGBoost parameters."""
    params = dict(option_pair(option) for option in passthrough) 

    if (unknown := sorted(set(params) - xgboost_names())):
        raise click.BadParameter(
            f"Neither this command nor XGBoost has key(s) `{ '`, `'.join(unknown) }`. "
        )

    return params


def overrides(config: Config, fields: dict, tree_config: dict) -> dict:
    """The fields the command line set, to override into `config`."""
    context = click.get_current_context()

    given = { name: value for name, value in fields.items()
              if context.get_parameter_source(name) 
                is not click.ParameterSource.DEFAULT }

    given["tree_config"] = config.tree_config | tree_config

    return given


@click.command(cls = Passthrough,
               context_settings = { "ignore_unknown_options": True })
@click.argument("outfile", type = click.Path(dir_okay = False))
@click.option("-c", "--config", "config_spec",
               default = CONFIG_DEFAULT, show_default = True,
              help = "Config to train, by name or as a `module:Class` path.")
@click.option("--config-file", type = click.Path(exists = True, dir_okay = False),
              help = "Config JSON to train (supersedes `--config`).")
@click.option("--n-steps", type = int, help = "Flow-matching time steps.")
@click.option("--n-events", type = EVENTS,
              help = f"Events to train on, or `{ ALL_EVENTS }` for entire dataset.")
@click.option("--seed", type = int, help = "Seed for the noise.")
@click.option("--s1phi", type = bool,
              help = "Encode phi as (sin, cos) rather than as one channel.")
@click.option("--multi-output", type = bool,
              help = "Use one multi-output BDT per channel group.")
@click.argument("passthrough", nargs = -1, type = click.UNPROCESSED,
                metavar = "[--XGBKEY=VALUE]...")
def train(outfile: str, config_spec: str, config_file: str | None,
          passthrough: tuple[str, ...], **fields) -> None:
    """Train a FlowBDT model and export it to OUTFILE, which can be read by 
    `puppibuff.from_zip` or the CLI. 
    
    Trains --config, or --config-file with every other option applied on top.
    Pass any `--key=value` after OUTFILE as XGBoost parameter for the tree
    configuration.
    """
    from ..configs import Config
    from dataclasses import replace

    config = (Config.from_json(config_file) if config_file is not None
              else resolve_config(config_spec)())

    if outfile.startswith("-"):         # Took a passthrough option
        raise click.BadParameter(
            f"Pass OUTFILE before any `--XGBkey=value`, got { outfile !r}."
        )

    tree_config = tree_params(passthrough)

    config = replace(config, **overrides(config, fields, tree_config))
                                        # `[1:]` to drop the dataset
    codec, model, x, y = config.setup()[1:]

    model.fit(x, y)

    to_zip(outfile, config, codec, model)

    click.echo(f"Wrote config, codec and model to { outfile }.")
