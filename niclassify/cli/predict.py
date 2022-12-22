from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from .enums import TaxonomicHierarchy, Methods


def _predict(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input data containing features on which to predict",
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
    classifier: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="A trained classifier model to use for making predictions",
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
    columns: List[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Feature column names or slices, separated by a comma (e.g. aa_dist,ks_dist:ka_dist,1:5). Column numbers are 0-indexed. May be specified multiple times for a combination",
        show_default=False,
    ),
    columns_file: Optional[Path] = typer.Option(
        None,
        "--columns-selection",
        "-s",
        help="A file specifying selected feature columns, with each column name on a new line. --columns not required if specified.",
        prompt=True,
        show_default=False,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=False,
        resolve_path=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output classifier archive.",
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
    label_column: str = typer.Option(
        ...,
        "--label",
        "-l",
        help="Name of label column containing known status labels. Must not be contained in training column selection.",
        prompt=True,
        show_default=False,
        rich_help_panel="Requirements",
    ),
):
    """
    Make predictions on a set of features using a trained classifier model.

    Options marked [red]\[required][/] will be prompted for if not provided. One of either --columns or --columns-selection is required.
    """
    print(locals())
