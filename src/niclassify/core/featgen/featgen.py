import itertools
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from Bio import AlignIO
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align.analysis import calculate_dn_ds_matrix
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceMatrix
from Bio.SeqRecord import SeqRecord

from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data

# TODO: rewrite to handle larger-than-memory files


def get_statistics(
    dist_matrix: DistanceMatrix, delimitation: pl.LazyFrame, distname: str
) -> pl.LazyFrame:
    """Calculate distance statistics from a given DistanceMatrix."""
    max_len = max(len(row) for row in cast(list[list[int]], dist_matrix.matrix))
    full_matrix = np.zeros((max_len, max_len))
    flat = np.array(list(itertools.chain(*cast(list[list[int]], dist_matrix.matrix))))
    full_matrix[np.tril_indices(max_len)] = flat
    full_matrix = full_matrix + full_matrix.T
    original_diagonals = flat[:: (max_len + 1)]
    np.fill_diagonal(full_matrix, original_diagonals)
    names_without_prefix = [
        name.partition("_")[2] for name in cast(list[str], dist_matrix.names)
    ]
    return (
        pl.from_numpy(
            full_matrix,
            # BUG This assumes a split name (double check if there's a nosplit prefix)
            schema=names_without_prefix,
            orient="row",
        )
        .lazy()
        .with_columns(pl.Series("Left", names_without_prefix))
        .unpivot(index="Left", variable_name="Right", value_name="dist")
        .join(
            delimitation.select(pl.col.UID, pl.col.delim_OTU.alias("Left_OTU")),
            left_on="Left",
            right_on="UID",
            how="left",
        )
        .join(
            delimitation.select(pl.col.UID, pl.col.delim_OTU.alias("Right_OTU")),
            left_on="Right",
            right_on="UID",
            how="left",
        )
        .group_by("Left")
        .agg(
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .sum()
            .alias(f"{distname}_within_sum"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .mean()
            .alias(f"{distname}_within_mean"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .median()
            .alias(f"{distname}_within_median"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .std()
            .alias(f"{distname}_within_std"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .min()
            .alias(f"{distname}_within_min"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .max()
            .alias(f"{distname}_within_max"),
            pl.col.dist.filter(pl.col.Left_OTU != pl.col.Right_OTU)
            .mean()
            .alias(f"{distname}_without_mean"),
            pl.col.dist.filter(pl.col.Left_OTU == pl.col.Right_OTU)
            .sum()
            .alias(f"{distname}_without_sum"),
            pl.col.dist.filter(pl.col.Left_OTU != pl.col.Right_OTU)
            .median()
            .alias(f"{distname}_without_median"),
            pl.col.dist.filter(pl.col.Left_OTU != pl.col.Right_OTU)
            .std()
            .alias(f"{distname}_without_std"),
            pl.col.dist.filter(pl.col.Left_OTU != pl.col.Right_OTU)
            .min()
            .alias(f"{distname}_without_min"),
            pl.col.dist.filter(pl.col.Left_OTU != pl.col.Right_OTU)
            .max()
            .alias(f"{distname}_without_max"),
        )
        .rename({"Left": "UID"})
    )


def generate_features(
    input_path: Path,
    fasta_path: Path,
    output_path: Path,
    handler: Handler,
) -> None:
    """Generate features from the given data."""
    data = read_data(input_path, handler)

    columns = data.collect_schema().names()

    if "UID" not in columns:
        handler.error(handler.prefab.ERR_MISSING_UID, abort=True)

    if "delim_OTU" not in columns:
        handler.error(handler.prefab.ERR_MISSING_DELIM, abort=True)

    with handler.spin() as spinner:
        # Read the aligned/trimmed FASTA
        with fasta_path.open() as input_fasta:
            task = spinner.add_task("Reading FASTA...", total=1)
            alignment = cast(
                MultipleSeqAlignment,
                AlignIO.read(input_fasta, format="fasta"),  # pyright:ignore[reportUnknownMemberType]
            )
            spinner.update(task, description="Reading FASTA...done.", completed=1)

        # Get an Amino Acid translation of the alignment
        task = spinner.add_task("Translating to Amino Acid...", total=1)
        alignment_aa = MultipleSeqAlignment(
            [
                SeqRecord(
                    record.seq.translate(table="Invertebrate Mitochondrial"),  # pyright:ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    id=record.id,
                )
                for record in alignment  # pyright:ignore[reportUnknownVariableType]
            ]
        )
        spinner.update(
            task, description="Translating to Amino Acid...done.", completed=1
        )
    with handler.progress() as status:
        task = status.add_task(description="Calculating distances...", total=3)

        # Calculate the hamming distance of dna and aa, and the dn_ds, for each pair

        dist_dna = DistanceCalculator().get_distance(alignment)  # pyright:ignore[reportUnknownMemberType]
        status.update(task, description="Calculating distances...(dna done)", advance=1)

        dist_aa = DistanceCalculator().get_distance(alignment_aa)  # pyright:ignore[reportUnknownMemberType]
        status.update(task, description="Calculating distances...(aa done)", advance=1)

        # Adding these to the MSA for dn/ds calc
        # alignment.coordinates = Alignment(alignment).coordinates
        # alignment.sequences = Alignment(alignment).sequences
        # FIX: Can't handle ambiguous codons?
        # dn_matrix, ds_matrix = calculate_dn_ds_matrix(
        #     alignment, codon_table="Invertebrate Mitochondrial"
        # )
        status.update(task, description="Calculating distances...done.", advance=1)

        # Create statistics about each distance for each sequence
        dist_tasks = {"dna": dist_dna, "aa": dist_aa}
        delimitations = data.select(pl.col.UID, pl.col.delim_OTU)
        task = status.add_task(
            "Calculating distance statistics...", total=len(dist_tasks)
        )

        for name, dist_matrix in dist_tasks.items():
            stats = get_statistics(dist_matrix, delimitations, name)

            data = data.join(stats, on="UID", how="left")
            status.update(
                task,
                description=f"Calculating distance statistics...({name} done)",
                advance=1,
            )

        status.update(
            task,
            description="Calculating distance statistics...done.",
            complete=len(dist_tasks),
        )

    with handler.spin() as spinner:
        # Write out the new data
        task = spinner.add_task("Writing output...", total=1)
        data.sink_csv(output_path, separator="\t")
        spinner.update(task, description="Writing output...done.", total=1)
