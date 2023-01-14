from pathlib import Path
import dask.dataframe as dd
import pandas as pd

from .query_bold import query_bold

from ..interfaces import Handler
from ..utils import read_data


def identify(
    input_file: Path,
    output: Path,
    min_similarity: float,
    min_agreement: float,
    handler: Handler,
) -> None:
    handler.log("Attempting to identify sequences of unknown species...")

    # TODO add spinner
    data = read_data(input_file)

    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN)
        return

    successful_identification_count = {"count": 0}

    def count_assign(counter: int, row):
        # TODO check: this should work but apparently it's type: float so something's up
        if row["species_name"].isnull():
            identification = query_bold(
                row["nucleotides"], min_similarity, min_agreement, handler
            )
            if identification is not None:
                counter["count"] += 1
            return identification
        return row["species_name"]

    data["species_name"] = data.apply(
        lambda row: count_assign(successful_identification_count, row),
        axis=1,
        meta={column: "object" for column in data.columns},
    ).to_csv(output, single_file=True, index=False)
