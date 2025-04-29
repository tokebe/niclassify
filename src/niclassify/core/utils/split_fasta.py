# TODO: fasta path in, split fasta paths out
# Replace what's currently in trim_files, make it generic enough to be useable for delimit
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any

from Bio import SeqIO

from niclassify.core.interfaces.handler import Handler


def split_files(input_path: Path, handler: Handler) -> tuple[int, dict[str, Path]]:

    split_files: dict[str, Any] = {}
    try:
        n_seq = 0
        with (
            open(input_path, "r", encoding="utf8") as input_file,
            handler.spin() as spinner,
        ):
            task = spinner.add_task("Reading FASTA...", total=1)
            for record in SeqIO.parse(input_file, format="fasta"):
                n_seq += 1
                match = re.findall(r"^([^_]+)_(.*)", record.id)
                if len(match) == 0:
                    handler.error(
                        f"Sequence record {record.id} appears to have no split name. Make sure you chose the right split option for your data.",
                        abort=True,
                    )
                    return (0, {})
                split_name, _ = match[0]
                if split_name not in split_files:
                    split_files[split_name] = NamedTemporaryFile(
                        suffix=f"_{split_name}_smartsplit{input_path.suffix}",
                        mode="w",
                        encoding="utf8",
                        delete=False,
                    )

                split_files[split_name].write(f">{record.id}\n")
                split_files[split_name].write(f"{str(record.seq)}\n")
                spinner.update(
                    task,
                    description=f"Splitting FASTA...(wrote {n_seq} entries to {len(split_files.keys())} splits).",
                )
        pass
    finally:
        for file in split_files.values():
            file.close()

    split_paths = {split: Path(file.name) for split, file in split_files.items()}

    spinner.update(
        task,
        description=f"Splitting FASTA...done (wrote {n_seq} entries to {len(split_files.keys())} splits).",
        completed=True,
    )

    handler.debug(f"Smart-split files:")
    for path in split_paths.values():
        handler.debug(str(path))
    return n_seq, split_paths
