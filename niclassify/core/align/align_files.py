from ..interfaces import Handler
from ..enums import TaxonomicHierarchy
from dask.dataframe import DataFrame
from typing import List, Optional
from tempfile import TemporaryFile
from threading import Lock
from multiprocessing import cpu_count
from pathlib import Path
import platform
import psutil
import re
from Bio.Align.Applications import MuscleCommandline
from tempfile import NamedTemporaryFile
import subprocess
from ..dynamic_pool import DynamicPool
import math
import os

PLATFORM = platform.system()


def align_files(
    output_file: Path,
    written_files: List[Path],
    handler: Handler,
    cores: int = cpu_count(),
    output_all=False,
):

    with handler.spin() as status:

        lock = Lock()

        def align_file(file):

            split = re.search("_([^_]+)_unaligned|$", file.stem)[1]

            with lock:
                task = status.add_task(description=f"Aligning {split}...", total=1)

            muscle_exec = {
                "Windows": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86win32.exe",
                "Linux": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86linux64",
                "Darwin": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86darwin64",
            }[PLATFORM]

            if output_all:
                output_part = (
                    file.parent / f"{output_file.stem}_{split}_aligned{file.suffix}"
                )
            else:
                output_part = NamedTemporaryFile(
                    suffix=f"_{split}_aligned{file.suffix}",
                    mode="w",
                    encoding="utf8",
                    delete=False,
                )
                output_part.close()
                output_part = output_part.name

            alignment_call = f'{muscle_exec} -in "{file}" -out "{output_part}"'

            try:
                result = subprocess.run(
                    alignment_call, capture_output=True, check=True
                )
                with handler.debug_lock:
                    handler.debug(f"  Command for {split} alignment:")
                    handler.debug(f"  {alignment_call}")
            except subprocess.CalledProcessError as error:
                with handler.debug_lock:
                    handler.debug(f"  Command for {split} alignment:")
                    handler.debug(f"  {alignment_call}")
                    handler.debug(f"  stdout of {split} alignment:")
                    handler.debug(f"  {error.stdout}")
                    handler.debug(f"  stderr of {split} alignment:")
                    handler.debug(f"  {error.stderr}")
                    handler.error(
                        f"  An error occurred during alignment of {split}.",
                        "Additional details in above debug logs.",
                        abort=True,
                    )

            with lock:
                status.update(task, description=f"Aligning {split}...done.", advance=1)
            return Path(output_part)

        pool = DynamicPool(pool_type="thread")

        # assume processing takes 100x space to align
        tasks = [
            (align_file, math.ceil(file.stat().st_size / 1e4), (file,))
            for file in written_files
        ]

        output_parts = pool.map(tasks)

    with handler.progress(percent=True) as status:
        lock = Lock()
        task = status.add_task(
            description="Writing final output", total=len(output_parts)
        )

        with open(output_file, "w", encoding="utf8") as combined_output:
            for output_part in output_parts:
                with open(output_part, "r", encoding="utf8") as part:
                    for line in part:
                        combined_output.write(line)
                if not output_all:
                    os.unlink(output_part)
                with lock:
                    status.advance(task)
