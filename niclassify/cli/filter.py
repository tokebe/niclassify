from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum

def _filter(
    input_file: List[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Data to be filtered. Can be used multiple times to add multiple files with same variables, which are merged in output.",
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
    marker_codes: Optional[str] = typer.Option(
        "COI-5P",
        "--marker-codes",
        "-m",
        help="Marker codes to keep, separated by a comma.",
    ),
    base_pairs: Optional[int] = typer.Option(
        350, "--base-pairs", "-b", help="Minimum base pair count allowed.", min=0
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file.",
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
    fasta_output: Optional[str] = typer.Option(
        ...,
        "--output-fasta",
        "-f",
        help="FASTA output file.",
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
    Filter input data and prepare a FASTA file of the kept sequences.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    print(locals())
