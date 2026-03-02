from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.featgen.featgen import generate_features
from niclassify.core.interfaces.handler import Handler


def cli_featgen(
    input_file: Annotated[
        Path,
        typer.Option(
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
    ],
    input_fasta: Annotated[
        Path,
        typer.Option(
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
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            ...,
            "--output",
            "-o",
            help="Output data with classification features.",
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
    r"""Generate training features using statistics about genetic distance.

    Requires a species delimitation column named [bold]delim_OTU[/] to be present in the data.

    The specified [italic]split_level[/] must be present in the data (for example, default order requires [bold]order[/]). If the appropriate column is not provided, you will be asked whether to continue or not.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output_file, abort=True)
    generate_features(input_file, input_fasta, output_file, handler)
