from pathlib import Path
from typing import Annotated

import typer

from niclassify.core.align.align import align
from niclassify.core.enums import TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler

# TODO add arguments to output all files (add documentation that it'll all output with prefixes)
# otherwise it's all tempfiles and only the one output
# add warning that it'll generate n files and list files which will be overwritten


def cli_align(  # noqa:PLR0913
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
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output aligned FASTA (.fasta) file.",
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
    split_level: Annotated[
        TaxonomicHierarchy,
        typer.Option(
            "--split-on",
            "-s",
            help="Taxonomic level on which to split data for computation",
            case_sensitive=False,
        ),
    ] = TaxonomicHierarchy.order,
    output_all: Annotated[
        bool,
        typer.Option(
            "--output-all",
            "-a",
            help="Output all FASTA (.fasta) files, aligned and unaligned, separately for each split.",
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
    """Generate an aligned FASTA file using MUSCLE.

    The specified [italic]split_level[/] must be present in the data (for example, default order requires [bold]order_name[/]). If the appropriate column is not provided, you will be asked whether to continue or not.

    If splitting occurs, the output file will be a single combined FASTA file, where each group is labeled by the split level, with each group being aligned, but with no guarantee groups are aligned to one another. This file will be useable by the other steps without modification.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output, abort=True)

    align(input_file, output, split_level, handler, output_all)
