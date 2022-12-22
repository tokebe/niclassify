from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum

app = typer.Typer(rich_markup_mode="rich")


class TaxonomicHierarchy(str, Enum):
    none = "none"
    phylum = "phylum"
    _class = "class"
    order = "order"
    family = "family"
    subfamily = "subfamily"
    genus = "genus"

class Methods(str, Enum):
    bPTP = "bPTP"
    GMYC = "GMYC"


@app.command()
def search(
    geography: str = typer.Option(
        ...,
        "--geography",
        "-g",
        help="geographic location (e.g. Massachusetts).",
        prompt=True,
        show_default=False,
        rich_help_panel="Requirements",
    ),
    taxonomy: str = typer.Option(
        ...,
        "--taxonomy",
        "-t",
        help="Taxonomic label (e.g. hemiptera).",
        prompt=True,
        show_default=False,
        rich_help_panel="Requirements",
    ),
    # TODO implement an automatic path completion
    # see https://typer.tiangolo.com/tutorial/options-autocompletion/
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file.",
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
):
    """
    Search for sequence data from Barcode of Life Data System.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    print(locals())


@app.command()
def filter(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Data to be filtered.",
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
    marker_codes: Optional[str] = typer.Option(
        "COI-5P",
        "--marker-codes",
        "-m",
        help="Marker codes to keep, separated by a comma.",
    ),
    base_pairs: Optional[int] = typer.Option(
        350, "--base-pairs", "-b", help="Minimum base pair count allowed.", min=0
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file.",
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
    fasta_output: Optional[str] = typer.Option(
        ...,
        "--output-fasta",
        "-f",
        help="FASTA output file.",
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
):
    """
    Filter input data and prepare a FASTA file of the kept sequences.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    print(locals())


@app.command()
def align(
    input_fasta: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input FASTA file.",
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
        help="Output file.",
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
):
    """
    Align a given FASTA file using MUSCLE.

    Options marked [red]\[required][/] will be prompted for if not provided.
    """
    if input_fasta is None:
        input_fasta = typer.prompt("Input FASTA file")
    if output is None:
        output = typer.prompt("Output FASTA file")
    if split_level is None:
        split_level = typer.prompt("Taxonomic level to split data (e.g. order)")
    print(input_fasta, output, split_level)


@app.command()
def delimit(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing sample IDs.",
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
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output data with added species delimitation.",
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
    # TODO make this have an enum of possible values
    method: Methods = typer.Option(
        "bPTP", "--method", "-m", help="Alignment method to use"
    ),
):
    """
    Automatically delimit species based on genetic distance, using bPTP or GMYC.
    """
    print(locals())


# @app.command
# def prepare

# @app.command
# def train

# @app.command
# def predict

# TODO a helper function to generate a column selection file?

# TODO figure out what to call the composition of all data prep command?

# TODO: add argument to prompt instead of defaults when using interactive mode
# maybe just --interactive?

# TODO accept multiple inputs

# TODO confirm overwrite if file exists

# TODO add project options

# TODO write full niclassify composition


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    # TODO documentation
    # TODO use environment variables to set arguments when composing commands
    if ctx.invoked_subcommand is not None:
        return
    print("default")


if __name__ == "__main__":
    app()
