from pathlib import Path

from niclassify.core.enums import TaxonomicHierarchy
from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data


def format_data(input_file: Path, output: Path, handler: Handler) -> None:
    """Take user input to assign standardized column names."""
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

    taxon_columns = {}
    if taxon_levels is not None:
        taxon_columns = {
            handler.select(
                f"Select the column containing {level} labels",
                options=[
                    col for col in list(data.columns) if col != nucleotides_column
                ],
            ): f"{level}_name"
            for level in taxon_levels
        }

    marker_codes = None
    if handler.confirm(
        "Does the data contain a column specifying marker codes (such as COI-5P)?"
    ):
        marker_codes = handler.select(
            "Select the column containing marker codes (such as COI-5P)",
            options=[
                col
                for col in list(data.columns)
                if col not in [nucleotides_column, *(taxon_columns.keys() or ())]
            ],
        )

    column_mapping = {
        nucleotides_column: "nucleotides",
        **taxon_columns,
    }
    if marker_codes:
        column_mapping[marker_codes] = "marker_codes"

    with handler.spin() as status:
        task = status.add_task(description="Writing new file...", total=1)

        data.rename(mapping=column_mapping).sink_csv(
            output,
            separator="\t",
        )

        status.update(task, description="Writing new file...done.", advance=1)

    handler.log("Finished formatting data.")
