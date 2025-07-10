import typer

from niclassify.cli.feature.column_select import cli_column_select
from niclassify.cli.feature.featgen import cli_featgen
from niclassify.cli.utils.general import NaturalOrderGroup

feature_command = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    no_args_is_help=True,
    help="Generate and select features for the classifier.",
)
featgen = feature_command.command(name="featgen")(cli_featgen)
column_select = feature_command.command(name="select")(cli_column_select)
