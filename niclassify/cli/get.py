from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from .handler import Handler
from ..core.get import get
from ..core.utils import confirm_overwrite


def _get(
    geography: str = typer.Option(
        ...,
        "--geography",
        "-g",
        help="geographic location (e.g. Massachusetts).",
        prompt=True,
        show_default=False,
        rich_help_panel="Requirements",
    ),
    taxonomy: str = typer.Option(
        ...,
        "--taxonomy",
        "-t",
        help="Taxonomic label (e.g. hemiptera).",
        prompt=True,
        show_default=False,
        rich_help_panel="Requirements",
    ),
    # TODO implement an automatic path completion
    # see https://typer.tiangolo.com/tutorial/options-autocompletion/
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output .tsv file.",
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
    Search for sequence data from Barcode of Life Data System.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    if confirm_overwrite(output, Handler, abort=True):
        get(geography, taxonomy, output, Handler)
