from pathlib import Path
import dask.dataframe as dd
import json

import multiprocessing


NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.json") as nansfile:
    NANS = json.load(nansfile)


def read_data(file: Path) -> dd.DataFrame:
    data = dd.read_csv(
        file,
        sample=1000000,
        sample_rows=100,
        assume_missing=True,
        na_values=NANS,
        keep_default_na=True,
        engine="python",
        sep=None,
        dtype="object",
    )

    # if there's one column, it probably read wrong somehow
    # otherwise if there *should* be one column, this should be fine
    if len(data.columns) < 2:
        data = dd.read_csv(
            file,
            sample=1000000,
            sample_rows=100,
            assume_missing=True,
            na_values=NANS,
            keep_default_na=True,
            engine="python",
            sep="\t",
            dtype="object",
        )

    core_count = multiprocessing.cpu_count()

    part_count = 0
    for part in data.partitions:
        part_count += 1

    if part_count < core_count:
        return data.repartition(npartitions=core_count)
    return data
