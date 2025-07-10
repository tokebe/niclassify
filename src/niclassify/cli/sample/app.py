import typer

from niclassify.cli.sample.filter import cli_filter
from niclassify.cli.sample.format import cli_format
from niclassify.cli.sample.get import cli_get
from niclassify.cli.sample.identify import cli_identify
from niclassify.cli.sample.lookup import cli_lookup
from niclassify.cli.utils.general import NaturalOrderGroup

sample_command = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    no_args_is_help=True,
    help="Operate on tab-delimited sample data.",
)
format = sample_command.command(name="format")(cli_format)
get = sample_command.command(name="get")(cli_get)
filter = sample_command.command(name="filter")(cli_filter)
identify = sample_command.command(name="identify")(cli_identify)
lookup = sample_command.command(name="lookup")(cli_lookup)
