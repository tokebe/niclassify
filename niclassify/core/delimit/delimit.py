from multiprocessing import cpu_count
from pathlib import Path
from niclassify.core.enums import Methods, TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler


def delimit(
    input_file: Path,
    output_file: Path,
    split_level: TaxonomicHierarchy,
    handler: Handler,
    cores: int = cpu_count()
):
    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    if "UID" not in data.columns:
        handler.error(handler.prefab.MISSING_UID, abort=True)

    if split_level != "none" and f"{split_level}_name" not in data.columns:
        handler.confirm(
            f"Column {split_level}_name not present in data. Continue without split?",
            abort=True,
        )

    if split_level != "none":
        splits = data[f"{split_level}_name"].unique().compute(num_workers=cores)
    else:
        splits = None

    # TODO: implement bPTP delimitation
    # make a distance matrix, then UPGMA tree, from fasta
    # run bPTP
