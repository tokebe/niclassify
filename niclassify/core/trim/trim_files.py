import os
from pathlib import Path
from typing import Any
import re
from tempfile import NamedTemporaryFile
from multiprocessing import cpu_count

from Bio import SeqIO

from niclassify.core.dynamic_pool import DynamicPool
from niclassify.core.interfaces.handler import Handler
from niclassify.core.trim import trim
from niclassify.core.utils.split_fasta import split_files

# TODO: add confirm_overwrites for output_all


def trim_files(
    input_path: Path,
    output_path: Path,
    handler: Handler,
    min_agreement,
    cores: int,
    output_all,
):

    n_seq, split_paths = split_files(input_path, handler)

    out_files = {
        split: (
            open(
                output_path.parent
                / f"{output_path.stem}_{split}_trim{output_path.suffix}",
                "w",
                encoding="utf8",
            )
            if output_all
            else NamedTemporaryFile(
                suffix=f"_{split}_trim{output_path.suffix}",
                mode="w",
                encoding="utf8",
                delete=False,
            )
        )
        for split in split_paths.keys()
    }

    for file in out_files.values():
        file.close()

    try:
        for split, split_path in split_paths.items():
            trim(
                split_path,
                Path(out_files[split].name),
                handler,
                min_agreement,
            )
            os.unlink(split_path)
        with handler.progress() as progress, open(output_path, "w") as output_file:
            task = progress.add_task("Writing final output", total=n_seq)
            for split, outfile in out_files.items():
                with open(outfile.name, "r") as file:
                    for line in file:
                        if line.startswith(">"):
                            progress.advance(task)
                        output_file.write(line)
                if not output_all:
                    os.unlink(outfile.name)

    finally:
        for file in out_files.values():
            file.close()

    handler.log(f"Wrote {n_seq} sequences to combined file.")
