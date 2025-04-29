from pathlib import Path
from typing import cast
import polars as pl

from niclassify.core.identify.query_bold import query_bold

from niclassify.core.interfaces import Handler
from niclassify.core.utils import read_data

from threading import Lock


def identify(
    input_file: Path,
    output_file: Path,
    min_similarity: float,
    min_agreement: float,
    handler: Handler,
) -> None:

    data = read_data(input_file)
    columns = data.collect_schema().names()

    if "UID" not in columns:
        handler.error(handler.prefab.ERR_MISSING_UID, abort=True)

    if "nucleotides" not in columns:
        handler.error(handler.prefab.ERR_MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    handler.log(
        "Attempting to identify sequences of unknown species (this will take some time)..."
    )

    if "order_name" in columns:
        orders = set(
            data.select(pl.col("order_name"))
            .unique()
            .collect(streaming=True)
            .to_series()
            .to_list()
        )
    else:
        orders = set()

    unknown_species: int = (
        (data.select(pl.col("species_name").null_count()).collect().item())
        if "species_name" in columns
        else data.select(pl.len()).collect(streaming=True).item()
    )

    global identified_count
    identified_count = 0
    global lock
    lock = Lock()

    with handler.progress(percent=True) as status:
        task = status.add_task(description="Querying BOLD", total=unknown_species)

        def count_assign(row):
            global identified_count
            global lock
            if row["species_name"] is not None:
                return dict(
                    subspecies_name=row.get("subspecies_name", None),
                    species_name=row["species_name"],
                    subgenus_name=row.get("subgenus_name", None),
                    genus_name=row.get("genus_name", None),
                    tribe_name=row.get("tribe_name", None),
                    subfamily_name=row.get("subfamily_name", None),
                    family_name=row.get("family_name", None),
                    order_name=row.get("order_name", None),
                    class_name=row.get("class_name", None),
                    phylum_name=row.get("phylum_name", None),
                )
            identification = query_bold(
                row["UID"],
                row["nucleotides"],
                min_similarity,
                min_agreement,
                orders,
                handler,
            )
            identification = cast(dict[str, str | None], identification)
            with lock:
                if identification["species_name"] is not None:
                    identified_count += 1
                status.advance(task)
            return identification

        data.with_columns(  # Insert taxon columns if they don't exist
            subspecies_name=pl.coalesce(pl.col("^subspecies_name$"), pl.lit(None)),
            species_name=pl.coalesce(pl.col("^species_name$"), pl.lit(None)),
            subgenus_name=pl.coalesce(pl.col("^subgenus_name$"), pl.lit(None)),
            genus_name=pl.coalesce(pl.col("^genus_name$"), pl.lit(None)),
            tribe_name=pl.coalesce(pl.col("^tribe_name$"), pl.lit(None)),
            subfamily_name=pl.coalesce(pl.col("^subfamily_name$"), pl.lit(None)),
            family_name=pl.coalesce(pl.col("^family_name$"), pl.lit(None)),
            order_name=pl.coalesce(pl.col("^order_name$"), pl.lit(None)),
            class_name=pl.coalesce(pl.col("^class_name$"), pl.lit(None)),
            phylum_name=pl.coalesce(pl.col("^phylum_name$"), pl.lit(None)),
        ).with_columns(
            pl.struct(
                "UID",
                "nucleotides",
                "subspecies_name",
                "species_name",
                "subgenus_name",
                "genus_name",
                "tribe_name",
                "subfamily_name",
                "family_name",
                "order_name",
                "class_name",
                "phylum_name",
            )
            .map_elements(
                count_assign,
                skip_nulls=False,
                strategy="threading",
                return_dtype=pl.Struct(
                    {
                        "subspecies_name": pl.String,
                        "species_name": pl.String,
                        "subgenus_name": pl.String,
                        "genus_name": pl.String,
                        "tribe_name": pl.String,
                        "subfamily_name": pl.String,
                        "family_name": pl.String,
                        "order_name": pl.String,
                        "class_name": pl.String,
                        "phylum_name": pl.String,
                    }
                ),
            )
            .alias("identify_output")
        ).drop(  # Previous values are kept in identtify_output so no loss
            "subspecies_name",
            "species_name",
            "subgenus_name",
            "genus_name",
            "tribe_name",
            "subfamily_name",
            "family_name",
            "order_name",
            "class_name",
            "phylum_name",
        ).unnest(
            "identify_output"
        ).sink_csv(
            output_file, separator="\t"
        )

    handler.log(f"Successfully identified {identified_count} species.")
