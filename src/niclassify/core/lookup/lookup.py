from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Lock

import polars as pl

from niclassify.core.interfaces.handler import Handler
from niclassify.core.lookup.get_status import get_status
from niclassify.core.utils.read_data import read_data

RESERVED_COLUMNS = {"gbif_status", "itis_status", "final_status"}


def lookup(
    input_file: Path,
    output_file: Path,
    geography: str,
    handler: Handler,
) -> None:
    """Look up the statuses of known samples given a reference geography."""
    data = read_data(input_file)
    columns = data.collect_schema().names()

    if "species" not in columns:
        handler.error(handler.prefab.ERR_NO_SPECIES_NAME, abort=True)

    if not RESERVED_COLUMNS.isdisjoint(set(columns)) and not handler.confirm(
        "columns ",
        f"{', '.join(RESERVED_COLUMNS.intersection(set(columns)))} ",
        "will be overwitten. Continue?",
    ):
        return

    handler.log("Looking up statuses for known species...")

    # get all species statuses, avoiding duplicates
    species_names = set(
        data.select(pl.col.species.drop_nulls().unique())
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )

    lock = Lock()
    found_statuses: dict[str, tuple[str | None, str | None, str | None]] = {}

    with handler.progress(percent=False) as status:
        task = status.add_task(
            description="Looking up status", total=len(species_names)
        )

        def assign_status(
            species_name: str,
        ) -> None:
            status_gbif, status_itis, combined = get_status(
                species_name, geography, handler
            )
            found_statuses[species_name] = (status_gbif, status_itis, combined)
            with lock:
                status.advance(task)

        with ThreadPool() as pool:
            pool.map(
                assign_status,
                list(species_names),
            )

    species_identified = [
        species_name
        for species_name, (_, _, status) in found_statuses.items()
        if status is not None
    ]

    data = data.with_columns(
        pl.col.species.replace(
            list(found_statuses.keys()),
            [status_gbif for status_gbif, _, _ in found_statuses.values()],
        ).alias("gbif_status"),
        pl.col.species.replace(
            list(found_statuses.keys()),
            [status_itis for _, status_itis, _ in found_statuses.values()],
        ).alias("itis_status"),
        pl.col.species.replace(
            list(found_statuses.keys()),
            [status for _, _, status in found_statuses.values()],
        ).alias("final_status"),
    ).sink_csv(output_file, separator="\t")

    handler.log(
        f"Successfully retrieved statuses for {len(species_identified)} species."
    )
