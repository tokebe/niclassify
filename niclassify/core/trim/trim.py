from pathlib import Path
from ..interfaces import Handler
from Bio import SeqIO
from tempfile import NamedTemporaryFile
import shutil
from collections import Counter
import os
from Bio.Seq import Seq
import math
import textwrap

# TODO add spinner


def trim(
    input_path: Path, output_path: Path, handler: Handler, min_agreement: float = 0.9
):
    flip: bool
    offset: int
    contaminant_sequences = set()

    with open(
        input_path, "r", encoding="utf8"
    ) as input_file, handler.spin() as spinner:
        task = spinner.add_task("Reading FASTA...", total=1)
        frames = Counter()

        n_seq = 0
        for record in SeqIO.parse(input_file, format="fasta"):
            n_seq += 1
            spinner.update(f"Reading FASTA...(read {n_seq} entries)")
            success = False
            # Offsets where -3 is flipped with offset 2
            # But positives work normally
            for offset in range(-3, 3):
                flip = False
                if offset < 0:
                    flip = True
                    offset = abs(offset) - 1
                seq = record.seq if not flip else record.seq.reverse_complement()
                seq = seq.replace("-", "N")
                seq = ("N" * offset) + seq
                seq = seq + ("N" * (3 - len(seq) % 3))
                test = Seq(seq).translate(table="Invertebrate Mitochondrial")
                if "*" not in test:
                    frames.update([(flip, offset)])
                    success = True
            if not success:
                contaminant_sequences.add(record.id)

        handler.debug(
            "Reading frame votes (offset, where negative is",
            "reverse-complement and offset by abs - 2):",
        )
        handler.debug(
            ", ".join(
                [f"{offset}:{count / n_seq:.2f}" for offset, count in frames.items()]
            )
        )
        if not any((count / n_seq >= min_agreement for count in frames.values())):
            handler.error(
                "Minimum reading frame offset agreement not met.",
                "Your sequences may be heavily contaminated.",
                abort=True,
            )

        # determine the best offset to use
        flip, offset = frames.most_common()[0][0]

        task = spinner.update(
            task,
            description=f"Reading FASTA...done (read {n_seq} entries).",
            completed=1,
        )

    n_written = 0

    with open(output_path, "w", encoding="utf8") as output_file, handler.progress(
        percent=True
    ) as status:
        task = status.add_task(description="Writing to output FASTA", total=n_seq)
        for record in SeqIO.parse(input_file, format="fasta"):
            if record.id in contaminant_sequences:
                continue
            seq = record.seq if not flip else record.seq.reverse_complement()
            seq = seq.replace("-", "N")
            seq = ("N" * offset) + seq
            seq = seq + ("N" * (3 - len(seq) % 3))
            test = Seq(seq).translate(table="Invertebrate Mitochondrial")
            if "*" in test:
                contaminant_sequences.add(record.id)
                status.advance(task)
                continue
            output_file.write(f">{record.id}\n")
            output_file.write(
                "\n".join((seq[i : 60 + i] for i in range(0, len(seq), 60))) + "\n"
            )
            n_written += 1
            status.advance(task)

    if len(contaminant_sequences) > 0:
        handler.warning(
            " ".join(
                [
                    f"{len(contaminant_sequences)}",
                    f"sequence{'s' if len(contaminant_sequences) > 1 else ''}",
                    "failed to match reading frame consensus",
                    "and were not included in output.",
                ]
            )
        )
        handler.warning("These invalid sequences are:")
        for seqID in contaminant_sequences:
            handler.warning(seqID)

    handler.log(
        "".join(
            [
                f"Wrote {n_written} sequences with reading frame offset {offset}",
                " (reverse-complement); " if flip else "; ",
                f"{frames.most_common()[0][1] / n_seq:.2f} agreement.",
            ]
        )
    )


"""
- all sequences *should* be in same reading frame
- idea: get supermajority in correct reading frame, mark rest as contaminant?
- default to 90% agreement
- try forward and reverse, reversal must have agreement
"""
