from .handler import CLIHandler
from multiprocessing import cpu_count
from ..core.write import write
from ..core.utils import read_data
import typer
from pathlib import Path
from ..core.enums import TaxonomicHierarchy



n_cpus = cpu_count()


def _write(
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
        help="Output FASTA (.fasta) file.",
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
        help="Taxonomic level on which to split data. Only used if --output-all is specified.",
        case_sensitive=False,
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
    handler = CLIHandler(pre_confirm=pre_confirm, debug=debug)

    handler.confirm_overwrite(output, abort=True)

    data = read_data(input_file)
    if split_level != "none":
        splits = data[f"{split_level}_name"].unique().compute(num_workers=cores)
    else:
        splits = None

    write(data, splits, output, split_level, handler, cores, output_all=True)
