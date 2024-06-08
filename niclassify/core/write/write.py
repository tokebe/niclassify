from pathlib import Path
from multiprocessing import cpu_count
from ..interfaces import Handler
from ..enums import TaxonomicHierarchy
from dask.dataframe.core import DataFrame
from typing import List
from tempfile import NamedTemporaryFile
from threading import Lock


def write(
    data: DataFrame,
    splits: List[str] | None,
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    cores: int = cpu_count(),
    output_all=False,
) -> List[Path]:
    row_count = data.shape[0].compute()

    if splits is None:
        splits = ["nosplit"]
    files = {
        split: (
            Lock(),
            (
                open(
                    output_file.parent
                    / f"{output_file.stem}_{split}_unaligned{output_file.suffix}",
                    "w",
                    encoding="utf8",
                )
                if output_all
                else NamedTemporaryFile(
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

            def write_fasta(row):
                if splits is None:
                    split_name = "nosplit"
                    lock, file = files["nosplit"]
                else:
                    split_name = row[f"{split_level.value}_name"]
                    lock, file = files[row[f"{split_level.value}_name"]]
                with lock:
                    if splits is None:
                        label = f">{row['UID']}"
                    else:
                        label = f">{row[f'{split_level.value}_name']}_{row['UID']}\n"
                    file.write(label)
                    file.write(f"{row['nucleotides']}\n")
                    if split_name not in entries_written:
                        entries_written[split_name] = 0
                        file_map[split_name] = file.name
                    entries_written[split_name] += 1
                status.advance(task)

            def compute_part(df: DataFrame):
                df.apply(write_fasta, axis=1)
                return df

            data.map_partitions(
                compute_part, meta={column: "object" for column in data.columns}
            ).compute(num_workers=cores)
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
                f"Split [bold]{split_name}[/] has insufficient samples to perform an alignment and will be skipped."
            )
    if ask:
        handler.confirm("Continue with skipped files? (Abort otherwise)", abort=True)

    keep_files = [
        file_map[split_name]
        for split_name, n_written in entries_written.items()
        if n_written > 1
    ]

    return [
        Path(file.name) for _, file in files.values() if file.name in keep_files
    ]
