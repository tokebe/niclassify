from pathlib import Path
from ..interfaces import Handler


def confirm_overwrite(file: Path, handler: Handler, abort=False) -> bool:
    if file.exists():
        return handler.confirm(
            f"File {file.absolute()} already exists. Overwrite?", abort=abort
        )
    return True
