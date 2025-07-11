import sys
from pathlib import Path
from typing import cast

import polars as pl

from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data

RESERVED_COLUMNS = [
    "UID",
]


def filter_samples(
    input_files: Path,
    output_file: Path,
    marker_codes: str,
    base_pairs: int,
    handler: Handler,
) -> None:
    """Filter given samples to only valid samples and output to given filepath."""
    data = read_data(input_files, handler)

    columns = data.collect_schema().names()

    if "nucleotides" not in columns:
        handler.error(handler.prefab.ERR_MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    before_rows = cast(int, data.select(pl.len()).collect(engine="streaming").item())

    with handler.spin() as status:
        task = status.add_task(
            f"Filtering data (checking {before_rows} rows)...", total=1
        )
        # Remove rows missing allowed marker_codes
        if "marker_codes" in columns:
            data = data.with_columns(pl.col.marker_codes.cast(pl.String, strict=False))
            for code in marker_codes.split(","):
                data = data.filter(pl.col.marker_codes.str.contains(code))

        # Remove rows with fewer than base_pairs count
        data = data.with_columns(
            pl.col.nucleotides.cast(pl.String, strict=False)
        ).filter(pl.col.nucleotides.str.len_chars() >= base_pairs)

        after_rows = cast(int, data.select(pl.len()).collect(engine="streaming").item())

        # Create a unique ID column
        data.drop(RESERVED_COLUMNS, strict=False).with_columns(
            pl.concat_str(
                [pl.lit("ID_"), pl.int_range(pl.len(), dtype=pl.UInt32)]
            ).alias("UID"),
        ).sink_csv(output_file, separator="\t")

        status.update(
            task,
            description=f"Filtering data (checking {before_rows} rows)...done.",
            advance=1,
        )

    handler.log(
        "File filtered successfully",
        f"(removed {before_rows - after_rows} rows).",
    )
