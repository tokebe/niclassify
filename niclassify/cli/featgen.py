import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from ..core.enums import TaxonomicHierarchy, Methods

def _featgen(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing species delimitations.",
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
    input_fasta: Path = typer.Option(
        ...,
        "--input-fasta",
        "-f",
        help="FASTA file of aligned sequences.",
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
        help="Output data with added species delimitation.",
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
    Generate training features using statistics about genetic distance.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    print(locals())
