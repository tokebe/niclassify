from pathlib import Path
from typing import cast

import polars as pl
import typer

from niclassify.core.align.align_files import align_files
from niclassify.core.enums import TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data
from niclassify.core.write.write import write


def align(
    input_file: Path,
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    output_all: bool = False,
) -> None:
    """Write the given set of samples to fasta and align the sequences."""
    data = read_data(input_file)

    columns = data.collect_schema().names()

    if "nuc" not in columns:
        handler.error(
            handler.prefab.ERR_MISSING_NUCLEOTIDES_COLUMN,
            abort=True,
        )
        return

    if "UID" not in columns:
        handler.error(
            handler.prefab.ERR_MISSING_UID,
            abort=True,
        )

    handler.log("Aligning sequences...")

    if split_level != "none" and f"{split_level}_name" not in columns:
        handler.confirm(
            f"Column {split_level}_name not present in data. Continue without split?",
            abort=True,
        )

    if split_level != "none":
        splits = set[str | None](
            data.select(pl.col(f"{split_level}_name").unique())
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )
        if None in splits:
            null_count = (
                data.select(pl.col(f"{split_level}_name").null_count())
                .collect(engine="streaming")
                .item()
            )
            handler.warning(
                f"{null_count} samples are missing a value for {split_level}_name.",
            )
            choice = handler.select(
                "How would you like to proceed?",
                [
                    "Collect these samples into their own split (may degrade training quality)",
                    "Filter these samples from the training data",
                    "Abort",
                ],
            )
            if "Collect" in choice:
                splits.discard(None)
                splits.add("Unknown")
                data = data.with_columns(
                    pl.col(f"{split_level}_name").fill_null("Unknown")
                )
            elif "Filter" in choice:
                splits.discard(None)
                data = data.filter(pl.col(f"{split_level}_name").is_not_null())
            else:
                raise typer.Abort()
        if output_all:
            handler.confirm(
                f"With output_all set, {(len(splits) * 2) + 1} files will be generated. Continue?",
                abort=True,
            )
            handler.confirm_multiple_overwrite(
                [
                    *[
                        output_file.parent
                        / f"{output_file.stem}_{split}_unaligned{output_file.suffix}"
                        for split in splits
                    ],
                    *[
                        output_file.parent
                        / f"{output_file.stem}_{split}_aligned{output_file.suffix}"
                        for split in splits
                    ],
                ],
                abort=True,
            )
    else:
        splits = None

    written_files = write(
        data,
        cast(set[str] | None, splits),
        output_file,
        split_level,
        handler,
        output_all=output_all,
    )
    align_files(output_file, written_files, handler, output_all=output_all)

    handler.log(
        "Finished alignment. Please review and edit as necessary before continuing."
    )
