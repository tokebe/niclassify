from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.identify.identify import identify
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()


def cli_identify( # noqa: PLR0913
    input_file: Annotated[
        Path,
        typer.Option(
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
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output (.tsv) data with any identifiable species added.",
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
    ],
    similarity: Annotated[
        float,
        typer.Option(
            "--min-similarity",
            "-s",
            help="Minimum similarity for a match to be considered, from 0 to 1.",
            min=0,
            max=1,
        ),
    ] = 1,
    agreement: Annotated[
        float,
        typer.Option(
            "--min-agreement",
            "-a",
            help="Minimum proportion of highest-similarity matches that must agree for successful identification (if multiple exceed minimum, highest proportion will be used).",
            min=0,
            max=1,
        ),
    ] = 1,
    pre_confirm: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Automatically confirm dialogs such as file overwrite confirmations.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Output debug logs to stdout.",
        ),
    ] = False,
) -> None:
    """Identify species by looking up sequences on the Barcode of Life Data System (WARNING: SLOW).

    Requires [bold]nucleotides[/] column with sequences.
    If [bold]species_name[/] column is provided, pre-identified species will be skipped.
    If [bold]order_name[/] column is provided, any mismatching orders will produce warnings.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output, abort=True)
    identify(input_file, output, similarity, agreement, handler)
