import math
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

import polars as pl
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from itaxotools import asapy

from niclassify.core.dynamic_pool import DynamicPool
from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data
from niclassify.core.utils.split_fasta import split_files

distance_calculator = DistanceCalculator("identity")
tree_constructor = DistanceTreeConstructor()


def delimit(  # noqa:PLR0913
    input_path: Path,
    input_fasta: Path,
    output_path: Path,
    split: bool,
    handler: Handler,
    output_all: bool = True,
) -> None:
    """Delimit samples into OTUs for use in feature generation."""
    data = read_data(input_path)
    columns = data.collect_schema().names()

    if "nuc" not in columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    if "UID" not in columns:
        handler.error(handler.prefab.MISSING_UID, abort=True)

    if split:
        _, split_paths = split_files(input_fasta, handler)
    else:
        split_paths = {"nosplit": input_fasta}

    if output_all and handler.confirm_overwrite(
        output_path.parent.joinpath(output_path.stem), abort=True
    ):
        shutil.rmtree(output_path.parent.joinpath(output_path.stem), ignore_errors=True)
        output_path.parent.joinpath(output_path.stem).mkdir()

    first_run = [True]
    lock = Lock()
    total_samples = [0]
    total_otus = [0]
    data_held = [data]

    with handler.progress() as progress:
        task = progress.add_task("Delimiting OTUs", total=len(split_paths))

        def delim_task(split_name: str, path: Path) -> None:
            analysis = asapy.PartitionAnalysis(str(path))
            analysis.launch()
            if output_all:
                try:
                    analysis.fetch(  # pyright:ignore[reportUnknownMemberType] Arg untyped
                        str(output_path.parent / output_path.stem / split_name)
                    )
                except Exception as e:
                    handler.error(e)
            if analysis.results is None:
                handler.error(
                    "ASAP delimitation failed for unknown reason.", abort=True
                )
                sys.exit(1)
            uid_to_otu_mapping = {
                # Cut out the split name so it's just the original UID
                # Have to strip because whitespace is left in on read
                name.replace(
                    f"{split_name}_" if split_name != "nosplit" else "", ""
                ).strip(): f"{split_name}_{otu.strip()}"
                for name, otu in dict[str, str](
                    read_data(
                        Path(f"{analysis.results}/{path.stem}.Partition_1.csv"),
                        handler,
                        has_header=False,
                    )
                    .collect(engine="streaming")
                    .iter_rows()
                ).items()
            }
            with lock:
                data_held[0] = data_held[0].with_columns(
                    pl.col("UID" if first_run[0] else "delim_OTU")
                    .replace(
                        list(uid_to_otu_mapping.keys()),
                        list(uid_to_otu_mapping.values()),
                    )
                    .alias("delim_OTU")
                )
                first_run[0] = False
                total_samples[0] = total_samples[0] + len(uid_to_otu_mapping)
                total_otus[0] = total_otus[0] + len(set(uid_to_otu_mapping.values()))
                progress.advance(task)

        pool = DynamicPool(pool_type="thread")

        # assume processing takes 5x space to delimit
        tasks: list[
            tuple[Callable[[str, Path], None], int, tuple[str, Path], dict[str, Any]]
        ] = [
            (
                delim_task,
                math.ceil(path.stat().st_size / 1e6) * 5,
                (split_name, path),
                {},
            )
            for split_name, path in split_paths.items()
        ]

        pool.map(tasks)

    with handler.spin() as status:
        task = status.add_task("Writing finished file...", total=1)
        data_held[0].sink_csv(output_path, separator="\t")
        status.update(task, completed=1, description="Writing finished file...done.")
    message = f"Delimited {total_samples[0]} samples into {total_otus[0]} OTUs"
    if len(split_paths) > 1:
        message += f" across {len(split_paths)} splits" if len(split_paths) > 1 else ""
    message += "."
    handler.log(message)
