from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.get.get import get
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()


def cli_get(
    geography: Annotated[
        str,
        typer.Option(
            "--geography",
            "-g",
            help="geographic location (e.g. Massachusetts).",
            prompt="geographic location (e.g. Massachusetts).",
            show_default=False,
            rich_help_panel="Requirements",
        ),
    ],
    taxonomy: Annotated[
        str,
        typer.Option(
            "--taxonomy",
            "-t",
            help="Taxonomic label (e.g. hemiptera).",
            prompt="Taxonomic label (e.g. hemiptera).",
            show_default=False,
            rich_help_panel="Requirements",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output (.tsv) file.",
            prompt="Output (.tsv) file",
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
    """Search for sequence data from Barcode of Life Data System.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output, abort=True)
    get(geography, taxonomy, output, handler)
