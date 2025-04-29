from multiprocessing import cpu_count
import typer
from pathlib import Path
from typing import List, Optional

from niclassify.core.trim.trim_files import trim_files
from niclassify.core.trim import trim
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()

def _trim(
    input_file: Path = typer.Option(
        ...,
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
    output_file: Path = typer.Option(
        ...,
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
    no_split: bool = typer.Option(
        False,
        "--no-split",
        "-s",
        help="Set if the input Aligned FASTA was generated without splits (see align help)"
    ),
    agreement: float = typer.Option(
        0.9,
        "--min-agreement",
        "-a",
        help="Minimum proportion of aligned sequences that must agree on a reading frame.",
        min=0,
        max=1,
    ),
    output_all: bool = typer.Option(
        False,
        "--output-all",
        "-a",
        help="Output all trimmed FASTA (.fasta) files separately for each split. Ignored if --no-split is set."
    ),
    cores: int = typer.Option(
        n_cpus,
        "--cores",
        "-c",
        help="Number of cores to use. Defaults to system core count (i.e. the default changes). Ignored if --no-split is set.",
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
    Trim the input FASTA (.fasta) to a reading frame containing no stop codons.

    The split level must match the previously used split level from alignment.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output_file, abort=True)
    if no_split:
        trim(input_file, output_file, handler, agreement)
    else:
        trim_files(input_file, output_file, handler, agreement, cores, output_all)
