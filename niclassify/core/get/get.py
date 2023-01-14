from .query_bold import query_bold
from .validate_file import validate_file
from pathlib import Path
from ...cli.handler import CLIHandler

# TODO spinners for making request and for validating


def get(geography: str, taxonomy: str, output: Path, handler: CLIHandler) -> None:
    handler.log(f"Searching for {geography} {taxonomy} from BOLD...")
    query_bold(geography, taxonomy, output, handler)
    validate_file(output, handler)
