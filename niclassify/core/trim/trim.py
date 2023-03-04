from pathlib import Path
from ..interfaces import Handler
from Bio import SeqIO
from tempfile import NamedTemporaryFile
import shutil
from collections import Counter
import os

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
            # Check through possible reading frames until a valid one is found
            for offset in range(-4, 4):
                flip = False
                if offset < 0:
                    flip = True
                    offset = abs(offset) - 1
                untrimmed_len = len(record.seq) - offset
                tail = untrimmed_len - (untrimmed_len % 3) + 1
                seq = record.seq if not flip else record.seq.reverse_complement()
                success = "*" not in seq[offset:tail].translate()
                if success:
                    frames.update([offset if not flip else -offset - 1])
                    break
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
                abort=True
            )

        # determine the best offset to use
        flip = False
        best_offset = frames.most_common(1)[0][0]
        if best_offset < 0:
            flip = True
            best_offset = abs(offset) - 1

        # TODO use a counter to turn frames into a count
        # TODO step 2: write out with reading frame fixed
        # for each sequence, do the offset and then test that it works
        # if it doesn't, and it's not already in contaminant sequences, add it
        # don't write out contaminantes, and report them in the end
        for record in SeqIO.parse(input_fasta, format="fasta"):
            seq = record.seq if not flip else record.seq.reverse_complement()
            untrimmed_len = len(seq) - best_offset
            tail = untrimmed_len - (untrimmed_len % 3) + 1
            trimmed = seq[best_offset:tail]
            if "*" in trimmed.translate():
                contaminant_sequences.add(seq.id)
                continue
            output_fasta.write()

    shutil.copyfile(tempfile_path, output)


"""
- all sequences *should* be in same reading frame
- idea: get supermajority in correct reading frame, mark rest as contaminant?
- default to 90% agreement
- try forward and reverse, reversal must have agreement
"""
