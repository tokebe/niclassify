import csv
from pathlib import Path

import polars as pl
import yaml

from niclassify.core.interfaces.handler import Handler

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.yaml") as nansfile:
    NANS = yaml.safe_load(nansfile)


def read_data(
    file: Path,
    handler: Handler | None = None,
    skip_rows: int = 0,
    has_header: bool = True,
) -> pl.LazyFrame:
    """Determine the dialect of a csv-like file and read it.

    Returns Polars LazyFrame.
    """
    try:
        # Determine separator
        with file.open("r") as datafile:
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(datafile.readline()).delimiter

        data = pl.scan_csv(
            file,
            separator=delimiter,
            quote_char=None,
            null_values=NANS,
            rechunk=True,
            infer_schema_length=0,  # just make it all string
            skip_rows=skip_rows,
            has_header=has_header,
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
