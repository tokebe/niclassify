from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from ..core.enums import TaxonomicHierarchy
from multiprocessing import cpu_count
from ..core.utils import confirm_overwrite
from .handler import CLIHandler


n_cpus = cpu_count()

# TODO add arguments to output all files (add documentation that it'll all output with prefixes)
# otherwise it's all tempfiles and only the one output
# add warning that it'll generate n files and list files which will be overwritten


def _align(
    input_file: Path = typer.Option(
        ...,
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
    output: Path = typer.Option(
        ...,
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
    split_level: TaxonomicHierarchy = typer.Option(
        "order",
        "--split-on",
        "-s",
        help="Taxonomic level on which to split data for computation",
        case_sensitive=False,
    ),
    output_all: bool = typer.Option(
        False,
        "--output-all",
        "-a",
        help="Output all FASTA (.fasta) files, aligned and unaligned, separately for each split."
    ),
    cores: int = typer.Option(
        n_cpus,
        "--cores",
        "-c",
        help="Number of cores to use. Defaults to system core count (i.e. the default changes).",
        min=1,
        max=n_cpus,
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
    Generate an aligned FASTA file using MUSCLE.

    The specified [italic]split_level[/] must be present in the data (for example, default order requires [bold]order_name[/]). If the appropriate column is not provided, you will be asked whether to continue or not.

    If splitting occurs, the output file will be a single combined FASTA file, where each group is labeled by the split level, with each group being aligned, but with no guarantee groups are aligned to one another. This file will be useable by the other steps without modification.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler(pre_confirm=pre_confirm, debug=debug)

    confirm_overwrite(output, handler, abort=True)
