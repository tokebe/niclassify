from pathlib import Path
from ..interfaces import Handler
from Bio import SeqIO
from tempfile import NamedTemporaryFile
import shutil
from collections import Counter
import os
from Bio.Seq import Seq
import math

# TODO add spinner


def trim(input_file: Path, output: Path, handler: Handler, min_agreement: int = 0.9):
    with open(input_file, "r", encoding="utf8") as input_fasta, NamedTemporaryFile(
        mode="r", encoding="utf8", delete=False
    ) as output_fasta:
        tempfile_path = Path(output_fasta.name)

        frames = Counter()
        contaminant_sequences = set()
        for record in SeqIO.parse(input_fasta, format="fasta"):
            success = False
            for offset in range(-4, 4):
                flip = False
                if offset < 0:
                    flip = True
                    offset = abs(offset) - 1
                seq = record.seq if not flip else record.seq.reverse_complement()
                seq = seq + ("N" * ((3 * math.ceil(len(seq) / 3)) - len(seq)))
                test = Seq(seq).translate(table="Invertebrate Mitochondrial")
                if "*" not in test:
                    frames.update([(flip, offset)])
                    success = True
            if not success:
                contaminant_sequences.add(record.id)

        votes = sum(frames)
        handler.debug(
            "Reading frame votes (offset, where negative is",
            "reverse-complement and offset by abs - 1):",
        )
        handler.debug(
            ", ".join(
                [f"{offset}:{count / votes:.2f}" for offset, count in frames.items()]
            )
        )
        if not any((count / votes >= min_agreement for count in frames.values())):
            os.unlink(tempfile_path)
            handler.error(
                "Minimum reading frame offset agreement not met.",
                "Your sequences may be heavily contaminated.",
                abort=True,
            )

        # determine the best offset to use
        flip, offset = frames.most_common(1)[0]

        # TODO use a counter to turn frames into a count
        # TODO step 2: write out with reading frame fixed
        # for each sequence, do the offset and then test that it works
        # if it doesn't, and it's not already in contaminant sequences, add it
        # don't write out contaminantes, and report them in the end
        wrong_frame_sequences = []
        for record in SeqIO.parse(input_fasta, format="fasta"):
            if record.id in contaminant_sequences:
                continue
            seq = record.seq if not flip else record.seq.reverse_complement()
            seq = seq + ("N" * ((3 * math.ceil(len(seq) / 3)) - len(seq)))
            test = Seq(seq).translate(table="Invertebrate Mitochondrial")
            if "*" not in test:
                wrong_frame_sequences.append(record.id)
                continue
            output_fasta.write(f">{record.id}\n")
            output_fasta.write(f"{seq}\n")

        # TODO warn user about contaminant sequences and wrong frame sequences

    shutil.copyfile(tempfile_path, output)
    os.unlink(tempfile_path)


"""
- all sequences *should* be in same reading frame
- idea: get supermajority in correct reading frame, mark rest as contaminant?
- default to 90% agreement
- try forward and reverse, reversal must have agreement
"""
