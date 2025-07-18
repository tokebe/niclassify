from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.delimit.delimit import delimit
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()


def cli_delimit(  # noqa:PLR0913
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input (.tsv) file.",
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
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output (.tsv) data with added species delimitation.",
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
    # TODO: find better methods to support?
    # method: Annotated[
    #     str, typer.Option("--method", "-m", help="Alignment method to use")
    # ] = "ASAP",
    no_split: Annotated[
        bool,
        typer.Option(
            "--no-split",
            "-s",
            help="Set if the input Aligned FASTA was generated without splits (see align help)",
        ),
    ] = False,
    output_all: Annotated[
        bool,
        typer.Option(
            "--output-all",
            "-a",
            help="Save all files from the species delimitation method. These will save in a folder matching the output file name.",
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
    r"""Automatically delimit species based on genetic distance, using bPTP or GMYC.

    The split level must match the previously used split level from alignment.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output_path, abort=True)
    delimit(input_file, input_fasta, output_path, (not no_split), handler, output_all)

    # delimit()
