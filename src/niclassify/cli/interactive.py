
from niclassify.core.interfaces.handler import Handler


def cli_interactive(handler: Handler) -> None:
    handler.log("Running NIClassify in interactive mode.\n")
