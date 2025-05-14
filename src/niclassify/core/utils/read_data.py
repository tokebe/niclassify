from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import yaml
import polars as pl

import multiprocessing
import csv

from niclassify.core.interfaces.handler import Handler


NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.yaml") as nansfile:
    NANS = yaml.safe_load(nansfile)


def read_data(file: Path, handler: Optional[Handler] = None) -> pl.LazyFrame:
    """Determine the dialect of a csv-like file and read it.

    Returns Polars LazyFrame.
    """

    try:
        # Determine separator
        with open(file, "r") as datafile:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(datafile.readline()).delimiter

        data = pl.scan_csv(
            file,
            separator=delimiter,
            quote_char=None,
            null_values=NANS,
            rechunk=True,
        )

        return data
    except Exception as error:
        if handler:
            handler.error(error)
            handler.error(
                f"There was an error reading the data at {file}. Please see the above error details for more information and ensure the integrity of your input files.",
                abort=True,
            )
        exit(1)  # Just here to keep the type checker happy
