from pathlib import Path
from ..interfaces import Handler
from ..utils import read_data
from ..enums import TaxonomicHierarchy
from multiprocessing import cpu_count


def format_data(
    input_file: Path, output: Path, handler: Handler, cores=cpu_count()
) -> None:
    # TODO marker_codes (or leave alone otherwise, not needed for filtering)
    data = read_data(input_file)

    nucleotides_column = handler.select(
        "Select the column containing nucleotide sequences",
        options=list(data.columns),
        abort=True,
    )

    taxon_levels = handler.select_multiple(
        "Which taxonomic hierarchy levels (if any) are present in the data?",
        options=[
            *[entry.value for entry in TaxonomicHierarchy if entry.value != "none"],
            "species",
        ],
        allow_empty=True,
    )

    taxon_columns = {
        handler.select(
            f"Select the column containing {level} labels",
            options=[col for col in list(data.columns) if col != nucleotides_column],
        ): f"{level}_name"
        for level in taxon_levels
    }

    marker_codes = handler.select(
        "Select the column containing marker codes such as COI-5P if present",
        options=[
            col
            for col in list(data.columns)
            if col not in [nucleotides_column, *taxon_columns.keys()]
        ],
        allow_empty=True,
    )

    column_mapping = {
        nucleotides_column: "nucleotides",
        marker_codes: "marker_codes",
        **taxon_columns,
    }

    with handler.spin() as status:

        task = status.add_task(description="Writing new file...", total=1)

        data.rename(columns=column_mapping).to_csv(
            output,
            single_file=True,
            index=False,
            sep="\t",
            compute_kwargs={"num_workers": cores},
        )

        status.update(task, description="Writing new file...done.", advance=1)

    handler.log("Finished formatting data.")
