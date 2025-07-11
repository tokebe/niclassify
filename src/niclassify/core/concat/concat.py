import sys
from pathlib import Path

import polars as pl
from rich.markup import escape

from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data


def concat(
    input_files: list[Path], output_file: Path, handler: Handler, diagonal: bool = False
) -> None:
    """Concatenate multiple TSV files together."""
    data_parts = [read_data(input_file, handler=handler) for input_file in input_files]

    try:
        with handler.spin() as spinner:
            task = spinner.add_task("Attempting to combine files...", total=1)
            pl.concat(
                data_parts,
                how="diagonal_relaxed" if diagonal else "vertical_relaxed",
                rechunk=True,
                parallel=True,
            ).sink_csv(output_file, separator="\t")
            spinner.update(
                task, description="Attempting to combine files...done.", complete=1
            )
        handler.log("Files combined successfully!")
    except Exception as error:
        handler.debug(escape(str(error)))
        handler.error(
            handler.prefab.ERR_TSV_CONCAT,
            abort=True,
        )
        sys.exit(1)
