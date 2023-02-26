from pathlib import Path
import dask.dataframe as dd
import pandas as pd

from ..interfaces import Handler
from ..utils import read_data
from .get_status import get_status
import numpy as np
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from itertools import repeat
from threading import Lock


def lookup(
    input_file: Path,
    output: Path,
    geography: str,
    handler: Handler,
    cores: int = cpu_count(),
) -> None:
    RESERVED_COLUMNS = {"gbif_status", "itis_status"}
    data = read_data(input_file)

    if "species_name" not in data.columns:
        handler.error(handler.prefab.NO_SPECIES_NAME)
        return

    if not RESERVED_COLUMNS.isdisjoint(set(data.columns)):
        if not handler.confirm(
            f"columns "
            f"{', '.join(RESERVED_COLUMNS.intersection(set(data.columns)))} "
            "will be overwitten. Continue?"
        ):
            return
    RESERVED_COLUMNS.add("final_status")  # won't be overwritten, just the null updated

    handler.log("Looking up statuses for known species...")

    # get all species statuses, avoiding duplicates
    species_names = data["species_name"].unique().dropna().compute(num_workers=cores)

    lock = Lock()

    with handler.progress(percent=False) as status:
        task = status.add_task(
            description="Looking up status", total=species_names.size
        )

        def get_status_with_progress(*args):
            status_info = get_status(*args)
            with lock:
                status.advance(task)
            return status_info

        with ThreadPool() as pool:
            species = pool.starmap(
                get_status_with_progress,
                zip(species_names, repeat(geography), repeat(handler)),
            )
    species_identified = [
        species_name
        for species_name, *_, final_status in species
        if final_status is not None
    ]

    statuses = pd.DataFrame(species).set_axis(
        [
            "species_name",
            "gbif_status",
            "itis_status",
            "final_status",
        ],
        axis="columns",
        copy=False,
    )

    def merge_statuses(df):
        with_new_statuses = df.drop(
            RESERVED_COLUMNS, axis="columns", errors="ignore"
        ).merge(statuses, how="left", on="species_name")
        missing_columns = 0
        for column in RESERVED_COLUMNS:
            if column not in df.columns:
                missing_columns += 1
                df[column] = with_new_statuses[column]
        if missing_columns < len(RESERVED_COLUMNS):
            df.update(with_new_statuses, overwrite=False)
        return df

    data.map_partitions(
        merge_statuses,
        meta={
            **{column: "object" for column in data.columns},
            **{
                column: "object"
                for column in RESERVED_COLUMNS
                if column not in data.columns
            },
        },
    ).to_csv(
        output,
        single_file=True,
        index=False,
        sep="\t",
        compute_kwargs={"num_workers": cores},
    )

    handler.log(
        f"Successfully retrieved statuses for {len(species_identified)} species."
    )
