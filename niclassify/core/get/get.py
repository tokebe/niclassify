from .bold_request import bold_request
from pathlib import Path
from typing import Callable

def get(geography: str, taxonomy: str, output: Path, handler: Callable):
    bold_request(geography, taxonomy, output, handler)
    validate_file(output)
