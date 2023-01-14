import asyncio
from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional, Union
from pathlib import Path
from enum import Enum
from ..core.lookup import lookup, get_geographies
from .validation import validate_geography
from .completion import complete_geography
from ..core.utils import confirm_overwrite
from .columnize import columnize
from .handler import CLIHandler


def list_geographies(value: bool):
    if value:
        columnize(get_geographies(), number=True)
        raise typer.Exit()


def _lookup(
    input_file: Path = typer.Option(
        None,
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
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output data with any known statuses added.",
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
    geography: str = typer.Option(
        None,
        "--geography",
        "-g",
        help="A reference geopgrahy with respect to which samples will be labeled as native or introduced.",
        show_default=False,
        show_choices=False,
        rich_help_panel="Requirements",
        callback=validate_geography,
        autocompletion=complete_geography,
    ),
    list_geographies: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all accepted geographies and exit.",
        callback=list_geographies,
        is_eager=True,
    ),
    pre_confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically confirm dialogs such as file overwrite confirmations.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Output debug logs to stdout.",
    ),
):
    """
    Establish any known species statuses as native or introduced using Global Biodiversity Information Facility and Integrated Taxonomic Information System.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler(debug=debug)
    geographies = get_geographies()
    # try to parse int input
    try:
        if geography is not None and int(geography) is not None:
            if int(geography) > 0 and int(geography) < len(geographies):
                geography = geographies[int(geography) - 1]
            else:
                raise typer.BadParameter(
                    "Geography must be exact match or integer index."
                )
    except ValueError:
        # Prompt user for a valid geography
        if geography is None:
            print(columnize(geographies, dry_run=True, number=True))
            index = 0
            while index < 1 or index > len(geographies) + 1:
                index = typer.prompt(
                    text="Reference geography for relative native/introduced labeling (input number)",
                    type=int,
                )
            geography = geographies[index - 1]

    if pre_confirm or confirm_overwrite(output, handler, abort=True):
        lookup(input_file, output, geography, handler)
