import asyncio
import typer
from typing import List, Optional, Union
from pathlib import Path
from enum import Enum
from ..core.lookup import lookup, get_geographies
from .validation import validate_geography
from .completion import complete_geography
from .columnize import columnize
from ..core.interfaces.handler import Handler
from multiprocessing import cpu_count

n_cpus = cpu_count()

# TODO use https://github.com/Exahilosys/survey or similar
# (select with filter) for better means of selecting
# a valid region

# TODO write gbif/itis native ranges as new columns
# If an unknown region is encountered, set it aside
# At end, report on unknown regions, tell user to report to github
# Ask if user would like to assign a status for those regions
# if so, run through regions, and back-assign those to related samples


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
    cores: int = typer.Option(
        n_cpus,
        "--cores",
        "-c",
        help="Number of cores to use. Defaults to system core count (i.e. the default changes).",
        min=1,
        max=n_cpus,
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

    Requires [bold]species_name[/] column.
    If [bold]final_status[/] column is provided, any existing statuses will be preserved.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    geographies = get_geographies()
    # try to parse int input
    try:
        if int(geography) is not None:
            if int(geography) > 0 and int(geography) < len(geographies):
                geography = geographies[int(geography) - 1]
            else:
                raise typer.BadParameter(
                    "Geography must be exact match or integer index."
                )
    except (ValueError, TypeError):
        # Prompt user for a valid geography
        if geography is None:
            geography = handler.select(
                "Reference geography for relative native/introduced labeling",
                geographies,
                abort=True,
            )

    handler.confirm_overwrite(output, abort=True)
    lookup(input_file, output, geography, handler, cores)
