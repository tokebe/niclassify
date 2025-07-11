from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.concat.concat import concat
from niclassify.core.interfaces.handler import Handler


def cli_concat(
    input_files: Annotated[
        list[Path],
        typer.Argument(
            help="Data to concatenate. Must have at least 2.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
            resolve_path=True,
            rich_help_panel="Requirements",
        ),
    ],
    output_file: Annotated[
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
    relaxed: Annotated[
        bool,
        typer.Option(
            "--relaxed",
            "-r",
            help="If set, combine even some files have columns that others do not, setting nulls where appropriate.",
        ),
    ] = False,
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
    """Concatenate multiple TSV sample files into one file.

    Useful to combine data obtained from BOLD with project data.

    Requires that each file is in the same format so columns may be joined appropriately.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)

    handler.confirm_overwrite(output_file, abort=True)
    concat(input_files, output_file, handler=handler, diagonal=relaxed)
