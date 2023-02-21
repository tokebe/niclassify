from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from .handler import CLIHandler
from ..core.get import get
from ..core.utils import confirm_overwrite

from multiprocessing import cpu_count

n_cpus = cpu_count()


def _get(
    geography: str = typer.Option(
        ...,
        "--geography",
        "-g",
        help="geographic location (e.g. Massachusetts).",
        prompt="geographic location (e.g. Massachusetts).",
        show_default=False,
        rich_help_panel="Requirements",
    ),
    taxonomy: str = typer.Option(
        ...,
        "--taxonomy",
        "-t",
        help="Taxonomic label (e.g. hemiptera).",
        prompt="Taxonomic label (e.g. hemiptera).",
        show_default=False,
        rich_help_panel="Requirements",
    ),
    # TODO implement an automatic path completion
    # see https://typer.tiangolo.com/tutorial/options-autocompletion/
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output (.tsv) file.",
        prompt="Output (.tsv) file.",
        show_default=False,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        writable=True,
        resolve_path=True,
        rich_help_panel="Requirements",
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
) -> None:
    """
    Search for sequence data from Barcode of Life Data System.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler(pre_confirm=pre_confirm, debug=debug)
    confirm_overwrite(output, handler, abort=True)
    get(geography, taxonomy, output, handler, cores)
