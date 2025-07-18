from collections import Counter
from pathlib import Path
from typing import cast

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord

from niclassify.core.interfaces.handler import Handler


def transform_sequence(
    record: SeqRecord, offset: int, flip: bool = False
) -> str | None:
    """Attempt to translate a sequence with a given offset and flip.

    Returns None if the record is invalid or fails translation.
    """
    if record.seq is None or record.id is None:
        return None
    seq = str(record.seq if not flip else record.seq.reverse_complement())
    seq = seq.replace("-", "N")

    # Pad beginning by offset, ensuring sequence length is multiple of 3
    seq = ("N" * offset) + seq
    seq = seq + ("N" * (3 - len(seq) % 3))

    try:
        test = Seq(seq).translate(table="Invertebrate Mitochondrial")
        if "*" not in test:
            return str(seq)
    except Exception:
        # Sequence had gaps, if these gaps cause errors, it's likely contaminant
        return None


def trim(
    input_path: Path, output_path: Path, handler: Handler, min_agreement: float = 0.9
) -> None:
    """Trim a FASTA file such that the most common reading frame among sequences is used.

    This generally follows this algorithm:
    1. For all sequences, obtain all possible valid reading frames (from possible offsets 1-3, and reverse complement offsets 1-3).
    2. For any sequences with no valid reading frame, discard them as invalid.
    3. Then, for each remaining sequence:
        If the sequence has the most common reading frame as a valid reading frame, use that offset.
        If the reading frame is different from the most common reading frame, discard the sequence as invalid
    4. What remains are the valid sequences all in the same reading frame.
    """
    flip: bool
    offset: int
    contaminant_sequences = set[str]()

    with (
        input_path.open(encoding="utf8") as input_file,
        handler.spin() as spinner,
    ):
        task = spinner.add_task("Reading FASTA...", total=1)
        frames = Counter[tuple[bool, int]]()

        n_seq = 0
        # Cast because SeqIO has some type weirdness
        for record in cast(FastaIterator, SeqIO.parse(input_file, format="fasta")):  # pyright:ignore[reportUnknownMemberType]
            n_seq += 1
            spinner.update(task, description=f"Reading FASTA...(read {n_seq} entries)")
            success = False
            # Offsets where -3 is flipped with offset 2
            # But positives work normally
            if record.seq is None or record.id is None:
                continue
            for offset in range(-3, 3):
                offset_val = offset
                flip = False
                if offset < 0:
                    flip = True
                    offset_val = abs(offset) - 1
                if transform_sequence(record, offset_val, flip) is not None:
                    frames.update([(flip, offset_val)])
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
        if not any(count / n_seq >= min_agreement for count in frames.values()):
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

    with (
        input_path.open(encoding="utf8") as input_file,
        output_path.open("w", encoding="utf8") as output_file,
        handler.progress(percent=True) as status,
    ):
        task = status.add_task(description="Writing to output FASTA", total=n_seq)
        for record in cast(FastaIterator, SeqIO.parse(input_file, format="fasta")):  # pyright:ignore[reportUnknownMemberType]
            if (
                record.seq is None
                or record.id is None
                or record.id in contaminant_sequences
            ):
                continue
            if (seq := transform_sequence(record, offset, flip)) is None:
                contaminant_sequences.add(record.id)
                status.advance(task)
                continue
            output_file.write(f">{record.id}\n")
            output_file.write(
                "\n".join(str(seq[i : 60 + i]) for i in range(0, len(seq), 60)) + "\n"
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
        for seq_id in contaminant_sequences:
            handler.warning(seq_id)

    handler.log(
        "".join(
            [
                f"Wrote {n_written} sequences with reading frame offset {offset}",
                " (reverse-complement); " if flip else "; ",
                f"{frames.most_common()[0][1] / n_seq:.2f} agreement.",
            ]
        )
    )
