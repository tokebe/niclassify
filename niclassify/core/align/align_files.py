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


PLATFORM = platform.system()


def align_files(
    output_file: Path,
    written_files: List[Path],
    splits: Optional[List[str]],
    split_level: TaxonomicHierarchy,
    handler: Handler,
    cores: int = cpu_count(),
    output_all=False,
):
    # TODO using multiprocessing pool, call MUSCLE
    # need to intelligently manage how many at a time using file size vs memory
    # run a number of files at a time such that the sum of squares of their sizes
    # leaves at least 2gb of memory to the system.
    # make sure to leave comments calling this "a naive approach" to avoiding
    # system OOM situations

    process_plan = []
    step = []
    max_safe_memory = int(psutil.virtual_memory().total - 2e9)  # leave 2GB to system
    step_size = 0
    for file in written_files:
        file_impact = file.stat().st_size ** 2
        if len(step) < 1:
            step.append(file)
            step_size += file_impact
            continue
        if step_size + file_impact <= max_safe_memory:
            step.append(file)
            step_size += file_impact
        else:
            process_plan.append(step)
            step = []
            step_size = 0

    with handler.spin() as status:

        def align_file(file):
            muscle_exec = {
                "Windows": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86win32.exe",
                "Linux": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86linux64",
                "Darwin": Path(__file__).parent.parent.parent
                / "bin/muscle3.8.31_i86darwin64",
            }[PLATFORM]

            if output_all:
                output_file = output_file.parent / f"{output_file.stem}_{re.search('_(.+)_unaligned|$', file.stem).group(1)}_unaligned.{output_file.suffix}"
            else:
                output_file = NamedTemporaryFile(suffix=f"_nosplit_unaligned.{output_file.suffix}",
                        mode="w",
                        encoding="utf8",
                        delete=False,)
                output_file.close()
                output_file = output_file.name

            alignment_call = MuscleCommandline(
                muscle_exec,
                input=file,
                out=output_file,
            )
    # task = status.add_task(description="Writing to FASTA", total=row_count)

        # TODO for each step, thread pool map to open subprocess with command
        # ensure each new one has a task started and closed when done
        # remember locks, ensure it works in parallel
        for step in process_plan:
            pass

        # TODO zip all the output files into the final file, going line by line to avoid reading all to memory
