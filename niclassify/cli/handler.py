from rich import print
import typer
import json
from pathlib import Path
import niclassify.core.interfaces.handler as handler
from contextlib import contextmanager
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)


# CONTEXT
CONTEXT = {"context": None}


class CLIHandler(handler.Handler):
    def __init__(self, pre_confirm: bool = False, debug: bool = False):
        self.pre_confirm = pre_confirm
        self._debug = debug

    def prefix_with_indent(*message, prefix: str) -> str:
        """Return message with prefix, respecting indent."""
        indent = len(message[0]) - len(message[0].lstrip())
        lstripped = " ".join(
            [part.lstrip() if i == 0 else part for i, part in enumerate(message)]
        )
        return f"{' ' * indent}{prefix} {lstripped}"

    def debug(self, *message: str):
        """Print message only if debugging is enabled."""
        if self._debug:
            self.log(
                f"[italic bright_black]{CLIHandler.prefix_with_indent(*message, prefix='DEBUG:')}[/]"
            )

    @staticmethod
    def log(*message: str):
        """Log a message."""
        if CONTEXT["context"] is not None:
            CONTEXT["context"].console.print(" ".join(message))
        else:
            print(" ".join(message))

    @staticmethod
    def message(*message: str):
        """Log a message and wait for the user to acknowledge."""
        CLIHandler.log(*message)
        typer.prompt("Enter to continue", hide_input=True)

    @staticmethod
    def warning(*message: str):
        """Log a message with a warning prefix to grab user attention."""
        CLIHandler.log(
            CLIHandler.prefix_with_indent(*message, prefix="[bold yellow]WARNING:[/]")
        )

    @staticmethod
    def error(*error: str, abort=False):
        """Log a message with an error prefix and exit if required."""
        CLIHandler.log(
            CLIHandler.prefix_with_indent(*error, prefix="[bold red]ERROR:[/]")
        )
        if abort:
            typer.Exit(code=1)

    def confirm(self, *message: str, abort=False, allow_pre_confirm=True):
        """Get a simply yes/no response from the user."""

        if (allow_pre_confirm and self.pre_confirm):
            CLIHandler.log(f"[italic bright_black]{' '.join(message)}: y[/]")
            return True

        return typer.confirm(
            ' '.join(message), abort=abort
        )

    @staticmethod
    @contextmanager
    def spin(transient=False):
        """Create a context which allows for one or more spinners to display that something is in progress."""
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

    @staticmethod
    @contextmanager
    def progress(transient=False, percent=False):
        """Create a context which allows for one or more spinners with progress bars."""

        if percent:
            progress_text = "{task.percentage:>3.0f}% | time remaining"
        else:
            progress_text = "{task.completed}/{task.total} | time remaining"

        if CONTEXT["context"] is not None:
            yield CONTEXT["context"]
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn(progress_text),
                TimeRemainingColumn(elapsed_when_finished=True),
                transient=transient
            ) as progress:
                CONTEXT["context"] = progress
                yield progress
        finally:
            CONTEXT["context"] = None
