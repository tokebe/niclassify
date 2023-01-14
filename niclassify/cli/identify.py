from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from ..core.utils import confirm_overwrite
from ..core.identify import identify

from .handler import CLIHandler


def _identify(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing sample sequences.",
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
        help="Output data with any identifiable species added.",
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
    similarity: float = typer.Option(
        float(1),
        "--min-similarity",
        "-s",
        help="Minimum similarity for a match to be considered, from 0 to 1.",
    ),
    agreement: float = typer.Option(
        float(1),
        "--min-agreement",
        "-a",
        help="Minimum proportion of highest-similarity matches that must agree for successful identification (if multiple exceed minimum, highest proportion will be used).",
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
    Identify species by looking up sequences on the Barcode of Life Data System (WARNING: SLOW).


    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler()
    if pre_confirm or confirm_overwrite(output, handler, abort=True):
        identify(input_file, output, similarity, agreement, handler)
