import typer
from pathlib import Path
from typing import List, Optional
from ..core.trim import trim
from .handler import CLIHandler

def _trim(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input (.fasta) file.",
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
        help="Output (.fasta) file, trimmed to proper reading frame.",
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
    agreement: float = typer.Option(
        0.9,
        "--min-agreement",
        "-a",
        help="Minimum proportion of aligned sequences that must agree on a reading frame.",
        min=0,
        max=1,
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
    Trim the input FASTA (.fasta) to a reading frame containing no stop codons.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output, abort=True)
    trim(input_file, output, handler, agreement)
