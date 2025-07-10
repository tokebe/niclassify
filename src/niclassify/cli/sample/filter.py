from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.filter.filter import filter_samples
from niclassify.core.interfaces.handler import Handler


def cli_filter( # noqa: PLR0913
    input_file: Annotated[
        list[Path],
        typer.Option(
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
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output (.tsv) data, filtered.",
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
    marker_codes: Annotated[
        str,
        typer.Option(
            "--marker-codes",
            "-m",
            help="Marker codes to keep, separated by a comma.",
        ),
    ] = "COI-5P",
    base_pairs: Annotated[
        int,
        typer.Option(
            "--base-pairs", "-b", help="Minimum base pair count allowed.", min=0
        ),
    ] = 350,
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
    """Filter input data. See options for default filtering cases.

    If [bold]marker_codes[/] column is provided, only allowed marker codes will be kept.
    If [bold]base_pairs[] column is provided, only sequences longer than --base-pairs will

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)

    handler.confirm_overwrite(output, abort=True)
    filter_samples(input_file, output, marker_codes, base_pairs, handler)
