from pathlib import Path
from typing import Annotated

import typer

from niclassify.cli.columnize import columnize
from niclassify.cli.completion.geography import complete_geography
from niclassify.cli.validation.geography import validate_geography
from niclassify.config.regions import REGIONS_FLAT
from niclassify.core.interfaces.handler import Handler
from niclassify.core.lookup.get_geographies import get_geographies
from niclassify.core.lookup.lookup import lookup


def list_geographies(value: bool) -> None:
    """Print all geographies to a list fitting the terminal."""
    if value:
        columnize(get_geographies(), number=True)
        raise typer.Exit()


def cli_lookup(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input data containing known species names. Must have column named [bold]species_name[/].",
            prompt=True,
            show_default=False,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
            resolve_path=True,
            rich_help_panel="Requirements",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output (.tsv) data with any known statuses added.",
            prompt=True,
            show_default=False,
            exists=False,
            file_okay=True,
            dir_okay=False,
            readable=False,
            writable=True,
            resolve_path=True,
            rich_help_panel="Requirements",
        ),
    ],
    geography: Annotated[
        str | None,
        typer.Option(
            "--geography",
            "-g",
            help="A reference geopgrahy with respect to which samples will be labeled as native or introduced. Can be the name or number from the geography list (see --list).",
            show_default=False,
            show_choices=False,
            rich_help_panel="Requirements",
            callback=validate_geography,
            autocompletion=complete_geography,
        ),
    ] = None,
    _list_geographies: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List all accepted geographies and exit.",
            callback=list_geographies,
            is_eager=True,
        ),
    ] = False,
    pre_confirm: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Automatically confirm dialogs such as file overwrite confirmations.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Output debug logs to stdout.",
        ),
    ] = False,
) -> None:
    """Establish any known species statuses as native or introduced using Global Biodiversity Information Facility and Integrated Taxonomic Information System.

    Requires [bold]species_name[/] column.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    geographies = get_geographies()
    # try to parse int input
    if not geography:
        geography = handler.select(
            "Reference geography for relative native/introduced labeling",
            geographies,
            abort=True,
        )
    try:
        selection = int(str(geography))
        if not 0 < selection < len(geographies):
            raise typer.BadParameter(
                "Geography must be exact name match or integer index."
            )
        geography = geographies[selection - 1]
    except ValueError:  # Geography is a string
        pass
    if geography not in REGIONS_FLAT:
        raise typer.BadParameter(
            "Geography not recognized. Please make an issue about this in the repository."
        )

    handler.confirm_overwrite(output, abort=True)
    lookup(input_file, output, str(geography), handler)
