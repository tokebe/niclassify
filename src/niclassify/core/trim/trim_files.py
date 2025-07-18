from pathlib import Path
from tempfile import NamedTemporaryFile

from niclassify.core.interfaces.handler import Handler
from niclassify.core.trim.trim import trim
from niclassify.core.utils.split_fasta import split_files


def trim_files(
    input_path: Path,
    output_path: Path,
    handler: Handler,
    min_agreement: float,
    output_all: bool,
) -> None:
    """Take a combined FASTA (which splits on a given rule) and trim its components."""
    n_seq, split_paths = split_files(input_path, handler)

    output_paths = dict[str, Path]()

    for split in split_paths:
        if output_all:
            output_paths[split] = (
                output_path.parent
                / f"{output_path.stem}_{split}_trim{output_path.suffix}"
            )
        else:
            with NamedTemporaryFile(
                suffix=f"_{split}_trim{output_path.suffix}",
                mode="w",
                encoding="utf8",
                delete=False,
            ) as file:
                output_paths[split] = Path(file.name)

    if output_all:
        handler.confirm_multiple_overwrite(list(output_paths.values()), abort=True)

    for split, split_path in split_paths.items():
        handler.log(f"Trimming split {split}...")
        trim(
            split_path,
            output_paths[split],
            handler,
            min_agreement,
        )
        split_path.unlink()
    with handler.progress() as progress, output_path.open("w") as output_file:
        task = progress.add_task("Writing final output", total=n_seq)
        for out_path in output_paths.values():
            with out_path.open("r") as file:
                for line in file:
                    if line.startswith(">"):
                        progress.advance(task)
                    output_file.write(line)
            if not output_all:
                out_path.unlink()

    handler.log(f"Wrote {n_seq} sequences to combined file.")
