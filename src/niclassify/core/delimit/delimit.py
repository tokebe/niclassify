from multiprocessing import cpu_count
from pathlib import Path
import re
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

from Bio import AlignIO, SeqIO
from niclassify.core.enums import Methods, TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

from niclassify.core.utils.read_data import read_data
from niclassify.core.utils.split_fasta import split_files
from bptp import run_bptp

distance_calculator = DistanceCalculator("identity")
tree_constructor = DistanceTreeConstructor()


# TODO: evaluate alternatives to bPTP
# - ABGD may work, can run purely binary, however you'll have to check licensing for redist
# - mPTP seems mildly promising, possibly an improvement over bPTP, same issue as above
# - Look into ASAP (Assemble Species by Automatic Partitioning)
#   https://github.com/iTaxoTools/ASAPy

# TODO add spinners/progress bars


def make_tree(fasta_path: Path, handler: Handler) -> str:
    """Read an aligned FASTA file and turn it into a UPGMA tree, in newick-string format."""
    with open(fasta_path, "r", encoding="utf8") as file:
        alignment = AlignIO.read(file, format="fasta")
    try:
        distance_matrix = distance_calculator.get_distance(alignment)
        upgma_tree = tree_constructor.upgma(distance_matrix)
        return upgma_tree.format("newick")
    except Exception as error:
        handler.error(error)
        handler.error(
            "Error constructing UPGMA tree for delimitation. See information above.",
            abort=True,
        )
        return ""


def delimit(
    input_path: Path,
    input_fasta: Path,
    output_path: Path,
    split: bool,
    handler: Handler,
    cores: int = cpu_count(),
):
    data = read_data(input_path)

    if "nucleotides" not in data.columns:
        handler.error(handler.prefab.MISSING_NUCLEOTIDES_COLUMN, abort=True)
        return

    if "UID" not in data.columns:
        handler.error(handler.prefab.MISSING_UID, abort=True)

    # TODO if no split, skip
    if split:
        _, split_paths = split_files(input_fasta, handler)
    else:
        split_paths = {"nosplit": input_fasta}

    # TODO: parallelize
    for path in split_paths.values():
        newick = make_tree(path, handler)
        match = re.findall(r".*_(.*)_smartsplit\..*", path.name)
        if len(match) == 0:
            handler.error(
                f"Error finding split in pathname {path} prior to delimitation. Make sure you chose the right split option for your data.",
                abort=True,
            )
            return
        split_name = match[0]
        with NamedTemporaryFile(suffix=f"{split_name}_newick", delete=False) as file:
            file.write(newick.encode("utf8"))
            newick_file = Path(file.name).resolve()
        with TemporaryDirectory() as tempdir:
            run_bptp(newick_file, (Path(tempdir) / "bptp_out"))
            print(tempdir)
            while True:
                pass
        os.unlink(newick_file)
            # Then grab the desired file from the tempdir and read to tsv output

    # Step 1: split the existing fasta into multiple fasta files

    # TODO: implement bPTP delimitation
    # make a distance matrix, then UPGMA tree, from fasta
    # run bPTP
