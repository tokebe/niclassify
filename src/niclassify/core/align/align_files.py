import sys
from niclassify.core.interfaces import Handler
from niclassify.core.enums import TaxonomicHierarchy
from typing import IO, List, Optional, cast
from tempfile import TemporaryFile
from threading import Lock
from multiprocessing import cpu_count
from pathlib import Path
import platform
import psutil
import re

# from Bio.Align.Applications import MuscleCommandline
from tempfile import NamedTemporaryFile
import subprocess
from niclassify.core.dynamic_pool import DynamicPool
import math
import os

PLATFORM = platform.system()
MUSCLE_EXEC = {
    "Windows": Path(__file__).parent.parent.parent / "bin/muscle_win",
    "Linux": Path(__file__).parent.parent.parent / "bin/muscle5_linux",
    "Darwin": Path(__file__).parent.parent.parent / "bin/muscle_macos",
}[PLATFORM]


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
            split = re.search("_([^_]+)_unaligned|$", file.stem)
            if split is not None:
                split = split[1]

            with lock:
                task = status.add_task(description=f"Aligning {split}...", total=1)

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

            alignment_call = [
                f"{MUSCLE_EXEC}",
                "-align" if "muscle5" in MUSCLE_EXEC.name else "-in",
                f"{file}",
                "-output" if "muscle5" in MUSCLE_EXEC.name else "-out",
                f"{output_part}",
            ]

            handler.debug(f"  Command for {split} alignment:")
            handler.debug(f"  {' '.join(alignment_call)}")
            process = subprocess.Popen(
                alignment_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Make and check buffer for updated status info, using the latest
            out = ""
            for char in iter(lambda: cast(IO[bytes], process.stderr).read(1), b""):
                out += char.decode(encoding="utf8")
                match = re.findall(r"(([0-9]+(\.[0-9]+)?%) ([\S ]+))[\r\n]", out)
                if len(match) == 0:
                    continue
                out = ""  # Discard excess buffer
                with lock:
                    status.update(
                        task, description=f"Aligning {split}...{match[-1][0]}"
                    )

            stdout, stderr = process.communicate()

            if process.returncode:
                handler.debug(f"  stdout of {split} alignment:")
                handler.debug(f"  {stdout.decode(encoding='utf8')}")
                handler.debug(f"  stderr of {split} alignment:")
                handler.debug(f"  {stderr.decode(encoding='utf8')}")
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
            (align_file, math.ceil(file.stat().st_size / 1e4), (file,), {})
            for file in written_files
        ]

        handler.log("\nAlignment provided by MUSCLE:\n")
        handler.log(
            subprocess.run([MUSCLE_EXEC, "--version"], capture_output=True)
            .stdout.decode(encoding="utf8")
            .removesuffix("\n\n")
        )
        handler.log("(C) Copyright 2004-2021 Robert C. Edgar.")
        handler.log("Redistributed for use in NIClassify under GPLv3.0")
        handler.log("See https://www.drive5.com/muscle/ for more info.")
        handler.log(
            'R.C. Edgar (2021) "MUSCLE v5 enables improved estimates of phylogenetic tree confidence by ensemble bootstrapping"'
        )
        handler.log(
            "https://www.biorxiv.org/content/10.1101/2021.06.20.449169v1.full.pdf\n"
        )

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
