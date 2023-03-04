import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from ..core.filter import filter_fasta
from .handler import CLIHandler


from multiprocessing import cpu_count

n_cpus = cpu_count()


def _filter(
    input_file: List[Path] = typer.Option(
        ...,
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
    output: Path = typer.Option(
        ...,
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
    marker_codes: Optional[str] = typer.Option(
        "COI-5P",
        "--marker-codes",
        "-m",
        help="Marker codes to keep, separated by a comma.",
    ),
    base_pairs: Optional[int] = typer.Option(
        350, "--base-pairs", "-b", help="Minimum base pair count allowed.", min=0
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
    Filter input data. See options for default filtering cases.

    If [bold]marker_codes[/] column is provided, only allowed marker codes will be kept.
    If [bold]base_pairs[] column is provided, only sequences longer than --base-pairs will

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = CLIHandler(pre_confirm=pre_confirm, debug=debug)

    handler.confirm_overwrite(output, abort=True)
    filter_fasta(
        input_file, output, marker_codes, base_pairs, handler, cores
    )
