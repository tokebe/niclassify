from pathlib import Path
from threading import Lock
from typing import Any

import polars as pl

from niclassify.config.general import CONFIG
from niclassify.core.identify.query_bold import query_bold
from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data


def identify(
    input_file: Path,
    output_file: Path,
    min_similarity: float,
    min_agreement: float,
    handler: Handler,
) -> None:
    """Attempt to identify samples using BOLD."""
    data = read_data(input_file)
    columns = data.collect_schema().names()

    if "UID" not in columns:
        handler.error(handler.prefab.ERR_MISSING_UID, abort=True)

    if "nuc" not in columns:
        handler.error(handler.prefab.ERR_MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    handler.log(
        "Attempting to identify sequences of unknown species (this will take some time)..."
    )

    if "order" in columns:
        orders = set[str](
            data.select(pl.col.order)
            .unique()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
    else:
        orders = set[str]()

    unknown_species: int = (
        (data.select(pl.col.species.null_count()).collect(engine="streaming").item())
        if "species" in columns
        else data.select(pl.len()).collect(engine="streaming").item()
    )

    identified_count = [0]
    lock = Lock()

    with handler.progress() as status:
        task = status.add_task(description="Querying BOLD", total=unknown_species)

        def count_assign(
            row: dict[str, Any], identified_count: list[int], lock: Lock
        ) -> dict[str, str | None]:
            if row["species"] is not None:
                return {name: row.get(name) for name in CONFIG.apis.bold.taxon_levels}
            identification = query_bold(
                row["UID"],
                row["nuc"],
                min_similarity,
                min_agreement,
                orders,
                handler,
            )
            with lock:
                if identification["species"] is not None:
                    # Using mutable to keep info because there isn't
                    # a more convenient alternative
                    identified_count[0] = identified_count[0] + 1
                status.update(
                    task,
                    advance=1,
                    description=f"Querying BOLD (Identified {identified_count[0]})",
                )
            return identification

        data.with_columns(  # Insert taxon columns if they don't exist
            **{
                name: pl.coalesce(pl.col(f"^{name}$"), pl.lit(None))
                for name in CONFIG.apis.bold.taxon_levels
            }
        ).with_columns(
            pl.struct("UID", "nuc", *CONFIG.apis.bold.taxon_levels)
            .map_elements(
                lambda row: count_assign(row, identified_count, lock),
                skip_nulls=False,
                strategy="threading",
                return_dtype=pl.Struct(
                    dict.fromkeys(CONFIG.apis.bold.taxon_levels, pl.String)
                ),
            )
            .alias("identify_output")
            # Previous values are kept in identify_output so no loss in dropping
        ).drop(*CONFIG.apis.bold.taxon_levels).unnest("identify_output").sink_csv(
            output_file, separator="\t"
        )

    handler.log(f"Successfully identified {identified_count[0]} species.")
