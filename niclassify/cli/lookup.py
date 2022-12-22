from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from .enums import TaxonomicHierarchy, Methods

def _lookup(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing known species names.",
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
        ...,
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
):
    """
    Establish any known species statuses as native or introduced using Global Biodiversity Information Facility and Integrated Taxonomic Information System.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    print("Not yet implemented!")
    # print(locals())
