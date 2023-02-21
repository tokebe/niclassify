from pathlib import Path
from typing import List
from ..interfaces import Handler
from multiprocessing import cpu_count
from ..utils import read_data
from dask import dataframe as dd

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
    data_parts = [read_data(inpu_file) for inpu_file in input_files]

    try:
        data = dd.concat(data_parts, axis="index", interleave_partitions=True)
    except ValueError as error:
        handler.debug(error)
        handler.error(
            "Failed to concatenate input files,",
            "they may be structurally incompatible.",
            "Run again with --debug to see exact error.",
            abort=True,
        )

    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN)
        return

    # Remove rows missing allowed marker_codes
    if "marker_codes" in data.columns:
        data["marker_codes"] = data["marker_codes"].astype(str)
        for code in marker_codes.split(","):
            data = data[data["marker_codes"].str.contains(code)]

    # Remove rows with fewer than base_pairs count
    data["nucleotides"] = data["nucleotides"].astype(str)
    data = data[data["nucleotides"].str.len() >= base_pairs]

    rows = data.shape[0].compute()

    lock = Lock()

    global count
    count = -1

    with handler.progress(percent=True) as status:
        task = status.add_task(description="Filtering rows", total=rows)

        def count_id(row, lock):
            with lock:
                global count
                count += 1
                status.advance(task)
                return f"ID_{count}"

        def make_ids(df, lock):
            uid = df.apply(count_id, lock, axis="columns", )
            df.insert(0, "UID", uid)
            return df

        # Create a unique ID column
        data.drop(columns=RESERVED_COLUMNS, errors="ignore").map_partitions(
            make_ids,
            lock,
            meta={"UID": "object", **{column: "object" for column in data.columns}},
        ).to_csv(
            output_file,
            single_file=True,
            index=False,
            sep="\t",
            compute_kwargs={"num_workers": cores},
        )
    handler.log(
        f"File{'s' if len(input_files) > 1 else ''} filtered successfully",
        f"(removed {rows - count+1} rows).",
    )
