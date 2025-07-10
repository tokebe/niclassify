from multiprocessing import cpu_count
from pathlib import Path

import typer

from niclassify.core.delimit.delimit import delimit
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()


def cli_delimit(
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
    input_fasta: Path = typer.Option(
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
    output_path: Path = typer.Option(
        ...,
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
    no_split: bool = typer.Option(
        False,
        "--no-split",
        "-s",
        help="Set if the input Aligned FASTA was generated without splits (see align help)",
    ),
    # TODO: find better methods to support?
    # method: Methods = typer.Option(
    #     "bPTP", "--method", "-m", help="Alignment method to use"
    # ),
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
    """Automatically delimit species based on genetic distance, using bPTP or GMYC.

    The split level must match the previously used split level from alignment.

    Options marked [red]\\[required][/] will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output_path, abort=True)
    delimit(input_file, input_fasta, output_path, (not no_split), handler, cores)

    # delimit()
