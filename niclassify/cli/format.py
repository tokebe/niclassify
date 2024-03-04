import typer
from multiprocessing import cpu_count
from ..core.format import format_data
from pathlib import Path
from ..core.interfaces.handler import Handler


n_cpus = cpu_count()


def _format(
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Input data (.tsv) to be reformatted",
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
        None,
        "--output",
        "-o",
        help="Output (.tsv) data with any known statuses added.",
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
    Create a new data file with NIClassify-compatible column names.

    Options in the 'Requirements' section will be prompted for if not provided.
    """
    handler = Handler(pre_confirm=pre_confirm, debug=debug)
    handler.confirm_overwrite(output, abort=True)
    format_data(input_file, output, handler, cores)
    pass
