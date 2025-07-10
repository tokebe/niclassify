from niclassify.core.interfaces.handler import Handler
from niclassify.core.lookup.combine_status import combine_status
from niclassify.core.lookup.query_gbif import query_gbif
from niclassify.core.lookup.query_itis import query_itis


def get_status(
    species_name: str, geography: str, handler: Handler
) -> tuple[str | None, str | None, str | None]:
    """Use GBIF and ITIS to determine whether a given species is native to the given reference geography."""
    status_gbif = query_gbif(species_name, geography, handler)
    status_itis = query_itis(species_name, geography, handler)
    combined = combine_status(status_gbif, status_itis)
    handler.log(
        "  {}: [bold]{}[/] (GBIF {} / ITIS {})".format(
            species_name,
            combined if combined is not None else "Unknown",
            status_gbif if status_gbif is not None else "Unknown",
            status_itis if status_itis is not None else "Unknown",
        )
    )
    return status_gbif, status_itis, combined
