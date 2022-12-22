from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from .enums import TaxonomicHierarchy

def _align(
    input_fasta: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input FASTA file.",
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
    split_level: TaxonomicHierarchy = typer.Option(
        "order",
        "--split-on",
        "-s",
        help="Taxonomic level on which to split data for computation",
        case_sensitive=False,
    ),
):
    """
    Align a given FASTA file using MUSCLE.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    if input_fasta is None:
        input_fasta = typer.prompt("Input FASTA file")
    if output is None:
        output = typer.prompt("Output FASTA file")
    if split_level is None:
        split_level = typer.prompt("Taxonomic level to split data (e.g. order)")
    print(input_fasta, output, split_level)
