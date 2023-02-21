from pathlib import Path
import dask.dataframe as dd
import pandas as pd

from .query_bold import query_bold

from ..interfaces import Handler
from ..utils import read_data

from multiprocessing import cpu_count
from threading import Lock


def identify(
    input_file: Path,
    output: Path,
    min_similarity: float,
    min_agreement: float,
    handler: Handler,
    cores: int = cpu_count(),
) -> None:

    data = read_data(input_file)

    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN)
        return

    handler.log("Attempting to identify sequences of unknown species...")

    if "order_name" in data.columns:
        orders = set(data["order_name"].unique().compute(num_workers=cores))
    else:
        orders = set()

    unknown_species: int = (
        data["species_name"].isnull().sum().compute(num_workers=cores)
    )

    global identified_count
    identified_count = 0
    lock = Lock()

    with handler.progress(percent=True) as status:
        task = status.add_task(description="Querying BOLD", total=unknown_species)

        def count_assign(row):
            global identified_count
            if pd.isnull(row["species_name"]):
                identification = query_bold(
                    row["nucleotides"], min_similarity, min_agreement, orders, handler
                )
                if identification is not None:
                    with lock:
                        identified_count += 1
                with lock:
                    status.advance(task)
                return pd.Series(identification, dtype="object")
            return pd.Series({"species_name": row["species_name"]}, dtype="object")

        def compute_part(df, partition_info=None):
            identification = df.apply(
                count_assign,
                axis=1,
            )
            df.update(identification)
            return df

        data.map_partitions(
            compute_part,
            meta={column: "object" for column in data.columns},
        ).to_csv(
            output,
            single_file=True,
            index=False,
            sep="\t",
            compute_kwargs={"num_workers": cores},
        )

    handler.log(f"Successfully identified {identified_count} species.")
