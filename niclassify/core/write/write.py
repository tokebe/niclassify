from pathlib import Path
from multiprocessing import cpu_count
from ..interfaces import Handler
from ..enums import TaxonomicHierarchy
from dask.dataframe import DataFrame
from typing import List
from tempfile import NamedTemporaryFile
from threading import Lock


def write(
    data: DataFrame,
    splits: List[str],
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    cores: int = cpu_count(),
    output_all=False,
) -> List[Path]:

    row_count = data.shape[0].compute()

    if output_all:
        if splits is None:
            files = {
                "file": (
                    Lock(),
                    open(
                        output_file.parent
                        / f"{output_file.stem}_nosplit_unaligned{output_file.suffix}",
                        "w",
                        encoding="utf8",
                    ),
                )
            }
        else:
            files = {
                split: (
                    Lock(),
                    open(
                        output_file.parent
                        / f"{output_file.stem}_{split}_unaligned{output_file.suffix}",
                        "w",
                        encoding="utf8",
                    ),
                )
                for split in splits
            }
    else:
        if splits is None:
            files = {
                "file": (
                    Lock(),
                    NamedTemporaryFile(
                        suffix=f"_nosplit_unaligned{output_file.suffix}",
                        mode="w",
                        encoding="utf8",
                        delete=False,
                    ),
                )
            }
        files = {
            split: (
                Lock(),
                NamedTemporaryFile(
                    suffix=f"_{split}_unaligned{output_file.suffix}",
                    mode="w",
                    encoding="utf8",
                    delete=False,
                ),
            )
            for split in splits
        }

    try:

        with handler.progress(percent=True) as status:
            task = status.add_task(description="Writing to FASTA", total=row_count)

            def write_fasta(row):
                if splits is None:
                    lock, file = files["file"]
                else:
                    lock, file = files[row[f"{split_level}_name"]]
                with lock:
                    file.write(f">{row[f'{split_level}_name']}_{row['UID']}\n")
                    file.write(f"{row['nucleotides']}\n")
                status.advance(task)

            def compute_part(df):
                df.apply(write_fasta, axis="columns")
                return df

            data.map_partitions(
                compute_part, meta={column: "object" for column in data.columns}
            ).compute(num_workers=cores)
    finally:
        for _, file in files.values():
            file.close()

    return [Path(file.name) for _, file in files.values()]
