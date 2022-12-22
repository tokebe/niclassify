from pathlib import Path

def confirm_overwrite(file: Path, handler, abort=False):
    if file.exists():
        return handler.confirm(f"File {file.absolute()} already exists. Overwrite?", abort=abort)
    return True
