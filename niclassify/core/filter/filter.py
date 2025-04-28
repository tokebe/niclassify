from pathlib import Path
from typing import List, cast
from niclassify.core.interfaces import Handler
from multiprocessing import cpu_count
from niclassify.core.utils import read_data
import polars as pl

from threading import Lock


RESERVED_COLUMNS = [
    "UID",
]


def filter_fasta(
    input_files: List[Path],
    output_file: Path,
    marker_codes: str,
    base_pairs: int,
    handler: Handler,
    cores: int = cpu_count(),
) -> None:
    data_parts = [read_data(input_file) for input_file in input_files]

    try:
        data = pl.concat(data_parts, how="vertical", rechunk=True)
    except ValueError as error:
        handler.debug(str(error))
        handler.error(
            handler.prefab.ERR_TSV_CONCAT,
            abort=True,
        )
        exit(1)

    columns = data.collect_schema().names()

    if "nucleotides" not in columns:
        handler.error(handler.prefab.ERR_MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    before_rows = cast(int, data.select(pl.len()).collect(streaming=True).item())

    # Remove rows missing allowed marker_codes
    if "marker_codes" in columns:
        data = data.with_columns(pl.col("marker_codes").cast(pl.String, strict=False))
        for code in marker_codes.split(","):
            data = data.filter(pl.col("marker_codes").str.contains(code))

    # Remove rows with fewer than base_pairs count
    data = data.with_columns(
        pl.col("nucleotides").cast(pl.String, strict=False)
    ).filter(pl.col("nucleotides").str.len_chars() >= base_pairs)

    after_rows = cast(int, data.select(pl.len()).collect(streaming=True).item())

    global lock
    lock = Lock()

    global count
    count = 0

    with handler.progress(percent=True) as status:
        task = status.add_task(description="Filtering rows", total=after_rows)

        def count_id(_):
            global lock
            global count
            with lock:
                count += 1
                status.advance(task)
                return f"ID_{count}"

        # Create a unique ID column
        data.drop(RESERVED_COLUMNS, strict=False).with_columns(
            pl.lit("").alias("UID"),
        ).select(
            pl.col("UID").map_elements(count_id, return_dtype=pl.String), pl.all()
        ).sink_csv(
            output_file, separator="\t"
        )

    handler.log(
        f"File{'s' if len(input_files) > 1 else ''} filtered successfully",
        f"(removed {before_rows - after_rows} rows).",
    )
