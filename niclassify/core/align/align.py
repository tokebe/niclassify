from pathlib import Path
from multiprocessing import cpu_count
from ..interfaces import Handler
from ..utils import read_data
from ..enums import TaxonomicHierarchy
from ..write import write
from .align_files import align_files


def align(
    input_file: Path,
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    cores: int = cpu_count(),
    output_all: bool = False,
):
    data = read_data(input_file)

    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    if "UID" not in data.columns:
        handler.error(handler.prefab.MISSING_UID, abort=True)

    handler.log("Aligning sequences...")

    if split_level != "none" and f"{split_level}_name" not in data.columns:
        handler.confirm(
            f"Column {split_level}_name not present in data. Continue without split?",
            abort=True,
        )

    if split_level != "none":
        splits = data[f"{split_level}_name"].unique().compute(num_workers=cores)
    else:
        splits = None

    if split_level != "none" and output_all:
        handler.confirm(
            f"With output_all set, {(len(splits) * 2) + 1} files will be generated. Continue?"
        )
        handler.confirm_multiple_overwrite(
            [
                output_file.parent
                / f"{output_file.stem}_{split}_unaligned.{output_file.suffix}"
                for split in splits
            ],
            abort=True,
        )
        handler.confirm_multiple_overwrite(
            [
                output_file.parent
                / f"{output_file.stem}_{split}_aligned.{output_file.suffix}"
                for split in splits
            ],
            abort=True,
        )

    written_files = write(
        data, splits, output_file, split_level, handler, cores, output_all=output_all
    )
    align_files(output_file, written_files, handler, cores, output_all=output_all)

    handler.log(
        "Finished alignment. Please review and edit as necessary before continuing."
    )
