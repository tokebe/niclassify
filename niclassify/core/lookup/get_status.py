from typing import Tuple, Optional

from ..interfaces.handler import Handler
from .query_gbif import query_gbif
from .query_itis import query_itis
from .combine_status import combine_status


def get_status(
    species_name: str, geography: str, handler: Handler
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    return species_name, status_gbif, status_itis, combined
