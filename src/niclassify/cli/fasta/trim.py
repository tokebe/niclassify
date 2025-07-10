from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.interfaces.handler import Handler
from niclassify.core.trim.trim import trim
from niclassify.core.trim.trim_files import trim_files


def cli_trim(  # noqa:PLR0913
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input aligned FASTA (.fasta) file.",
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
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output FASTA (.fasta) file, trimmed to proper reading frame.",
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
    no_split: Annotated[
        bool,
        typer.Option(
            False,
            "--no-split",
            "-s",
            help="Set if the input Aligned FASTA was generated without splits (see align help)",
        ),
    ] = False,
    agreement: Annotated[
        float,
        typer.Option(
            0.9,
            "--min-agreement",
            "-a",
            help="Minimum proportion of aligned sequences that must agree on a reading frame.",
            min=0,
            max=1,
        ),
    ] = 0,
    output_all: Annotated[
        bool,
        typer.Option(
            False,
            "--output-all",
            "-a",
            help="Output all trimmed FASTA (.fasta) files separately for each split. Ignored if --no-split is set.",
        ),
    ] = False,
    pre_confirm: Annotated[
        bool,
        typer.Option(
            False,
            "--yes",
            "-y",
            help="Automatically confirm dialogs such as file overwrite confirmations.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            False,
            "--debug",
            help="Output debug logs to stdout.",
        ),
    ] = False,
) -> None:
    """Trim the input FASTA (.fasta) to a reading frame containing no stop codons.

    The split level must match the previously used split level from alignment.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output_file, abort=True)
    if no_split:
        trim(input_file, output_file, handler, agreement)
    else:
        trim_files(input_file, output_file, handler, agreement, output_all)
