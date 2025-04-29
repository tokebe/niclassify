from niclassify.core.get.query_bold import query_bold
from niclassify.core.get.validate_file import validate_file
from pathlib import Path
from niclassify.core.interfaces.handler import Handler
from multiprocessing import cpu_count

# TODO spinners for making request and for validating


def get(
    geography: str,
    taxonomy: str,
    output: Path,
    handler: Handler,
) -> None:
    handler.log(f"Searching for {geography} {taxonomy} from BOLD...")
    query_bold(geography, taxonomy, output, handler)
    validate_file(output, handler)
