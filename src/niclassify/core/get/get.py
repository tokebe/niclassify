from pathlib import Path

from niclassify.core.get.query_bold import query_bold
from niclassify.core.get.validate_file import validate_file
from niclassify.core.interfaces.handler import Handler


def get(
    geography: str,
    taxonomy: str,
    output: Path,
    handler: Handler,
) -> None:
    """Query BOLD v4 for samples matching the search criteria."""
    handler.log(f"Searching for {geography} {taxonomy} from BOLD...")
    query_bold(geography, taxonomy, output, handler)
    validate_file(output, handler)
