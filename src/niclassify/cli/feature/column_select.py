from pathlib import Path

import typer
from rich import print


def cli_column_select(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing training features.",
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
        help="Output file containing column selections.",
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
    """Generate a text file containing column selections."""
    print(locals())
