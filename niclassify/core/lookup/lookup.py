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

RESERVED_COLUMNS = {"gbif_status", "itis_status", "final_status"}

# TODO more explicit branch logging


def lookup(input_file: Path, output: Path, geography: str, handler: Handler) -> None:
    handler.log("Looking up statuses for known species...")
    data = read_data(input_file)

    if "species_name" not in data.columns:
        handler.error(handler.prefab.NO_SPECIES_NAME)
        return

    if not RESERVED_COLUMNS.isdisjoint(set(data.columns)):
        if not handler.confirm(
            f"columns {', '.join(RESERVED_COLUMNS.intersection(set(data.columns)))} will be overwitten. Continue?"
        ):
            return

    species_names = data["species_name"].unique().dropna().compute()
    # get all species statuses, avoiding duplicates
    with ThreadPool() as pool:
        species = pool.starmap(get_status, zip(species_names, repeat(geography), repeat(handler)))
    # for species_name in species_names:
    #     if not pd.isna(species_name) and species_name not in species:
    #         species[species_name] = get_status(species_name, geography, handler)

    statuses = (
        pd.DataFrame(species)
        .set_axis(
            [
                "species_name",
                "gbif_status",
                "itis_status",
                "final_status",
            ],
            axis="columns",
            copy=False,
        )
    )
    statuses = dd.from_pandas(statuses, chunksize=1000000)
    # avoid duplicating column if it exists
    data.drop(RESERVED_COLUMNS, axis="columns", errors="ignore").merge(
        statuses, how="left", on="species_name"
    ).to_csv(output, single_file=True, index=False)
    # TODO figure out merging back into file and writing to output
