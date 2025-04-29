from niclassify.core.interfaces.handler import Handler
from niclassify.cli import _get
import typer


def _interactive(handler: Handler, cores: int) -> None:
    handler.log("Running NIClassify in interactive mode.\n")
