from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import polars as pl

from niclassify.core.enums import TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler


def write(  # noqa:PLR0913
    data: pl.LazyFrame,
    splits: set[str] | None,
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    output_all: bool = False,
) -> list[Path]:
    """Split the given data up and sequences out to FASTA files."""
    row_count = data.select(pl.len()).collect().item()

    if splits is None:
        splits = {"nosplit"}
    files = {
        split: (
            Lock(),
            (
                (
                    output_file.parent
                    / f"{output_file.stem}_{split}_unaligned{output_file.suffix}"
                ).open(
                    "w",
                    encoding="utf8",
                )
                if output_all
                else NamedTemporaryFile(  # noqa:SIM115 We're using a try:finally to ensure they close
                    suffix=f"_{split}_unaligned{output_file.suffix}",
                    mode="w",
                    encoding="utf8",
                    delete=False,
                )
            ),
        )
        for split in splits
    }

    entries_written: dict[str, int] = {}
    file_map: dict[str, str] = {}
    try:
        with handler.progress(percent=True) as status:
            task = status.add_task(description="Writing to FASTA", total=row_count)

            def write_fasta(row: tuple[str, str, str]) -> tuple[None, None, None]:
                uid, nucleotides, split_value = row
                if "nosplit" in splits:
                    split_name = "nosplit"
                    lock, file = files["nosplit"]
                else:
                    split_name = split_value
                    lock, file = files[split_value]
                with lock:
                    if "nosplit" in splits:
                        label = f">{uid}\n"
                    else:
                        label = f">{split_value}_{uid}\n"
                    file.write(label)
                    file.write(f"{nucleotides}\n")
                    if split_name not in entries_written:
                        entries_written[split_name] = 0
                        file_map[split_name] = file.name
                    entries_written[split_name] += 1
                status.advance(task)
                return None, None, None

            def compute_part(df: pl.DataFrame) -> pl.DataFrame:
                df.map_rows(write_fasta)
                return df

            data.select(
                pl.col.UID, pl.col.nucleotides, pl.col(f"{split_level.value}_name")
            ).map_batches(compute_part).collect(engine="streaming")

    finally:
        for _, file in files.values():
            file.close()

    ask = False
    for split_name, n_written in entries_written.items():
        if n_written <= 1:
            if split_name == "nosplit":
                handler.error(
                    "There are insufficient samples to perform an alignment. Alignment aborted.",
                    abort=True,
                )
            ask = True
            handler.warning(
                f"Split [bold]{split_name}[/] has insufficient samples ({n_written}) to perform an alignment and will be skipped."
            )
    if ask:
        handler.confirm("Continue with skipped files? (Abort otherwise)", abort=True)

    keep_files = [
        file_map[split_name]
        for split_name, n_written in entries_written.items()
        if n_written > 1
    ]

    return [Path(file.name) for _, file in files.values() if file.name in keep_files]
