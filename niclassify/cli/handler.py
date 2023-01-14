from rich import print
import typer
import json
from pathlib import Path
import niclassify.core.interfaces.handler as handler
from contextlib import contextmanager
from rich.progress import Progress, SpinnerColumn, TextColumn

# TODO figure out if you want special formatting/etc
# TODO create a CONTEXT manager that provides a spinner
# probably needs to be updatable, can set if transient? possibly etc.

# BUFFER
BUFFER = []
# CONTEXT
CONTEXT = {"context": None}

class CLIHandler(handler.Handler):

    def __init__(self, debug: bool = False):
        self.debug = debug

    def debug(self, message: str):
        if self.debug:
            self.log(f"[italics]DEBUG: {message}[/]")

    @staticmethod
    def log(message: str):
        if CONTEXT["context"] is not None:
            BUFFER.append({"func": CLIHandler.log, "args": [message]})
            return
        print(message)

    @staticmethod
    def message(message: str):
        if CONTEXT["context"] is not None:
            BUFFER.append({"func": CLIHandler.message, "args": [message]})
            return
        print(message)

    @staticmethod
    def warning(message: str):
        if CONTEXT["context"] is not None:
            BUFFER.append({"func": CLIHandler.warning, "args": [message]})
            return
        print(message)

    @staticmethod
    def error(error: str, should_exit=False):
        if CONTEXT["context"] is not None:
            BUFFER.append({"func": CLIHandler.log, "args": [error, should_exit]})
            return
        print(error)
        if should_exit:
            typer.Exit(code=1)

    @staticmethod
    def confirm(message: str, abort=False):
        return typer.confirm(message, abort=abort)

    @staticmethod
    @contextmanager
    def spin(transient=False):
        if CONTEXT["context"] is not None:
            yield CONTEXT["context"]
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=transient,
            ) as progress:
                CONTEXT["context"] = progress
                yield progress
        finally:
            CONTEXT["context"] = None
            while len(BUFFER) > 0:
                buffered_item = BUFFER.pop(0)
                buffered_item["func"](*buffered_item["args"])
