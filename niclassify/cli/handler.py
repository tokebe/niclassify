from rich import print
from rich.text import Text
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
from tempfile import NamedTemporaryFile
import re
from typing import List, Union
from .columnize import columnize
from threading import Lock
import atexit

# CONTEXT
CONTEXT = {"context": None}


# TODO create a log buffer that can be dumped to tempfile


class CLIHandler(handler.Handler):
    def __init__(self, pre_confirm: bool = False, debug: bool = False):
        self.pre_confirm = pre_confirm
        self._debug = debug
        self.debug_lock = Lock()
        self.logbuffer = []
        self.crashlog = None
        self.crashlog_lock = Lock()

    def prefix_with_indent(self, *message, prefix: str) -> str:
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
                f"[italic bright_black]{self.prefix_with_indent(*message, prefix='DEBUG:')}[/]"
            )

    def log(self, *message: str):
        """Log a message."""
        if CONTEXT["context"] is not None:
            CONTEXT["context"].console.print(" ".join(message))
        else:
            print(" ".join(message))
        self.logbuffer.append(" ".join(message))

    def message(self, *message: str):
        """Log a message and wait for the user to acknowledge."""
        self.log(*message)
        typer.prompt("Enter to continue", hide_input=True)

    def warning(self, *message: str):
        """Log a message with a warning prefix to grab user attention."""
        self.log(self.prefix_with_indent(*message, prefix="[bold yellow]WARNING:[/]"))

    def error(self, *error: str, abort=False):
        """Log a message with an error prefix and exit if required."""
        self.log(
            self.prefix_with_indent(
                *[str(e) for e in error], prefix="[bold red]ERROR:[/]"
            )
        )
        if abort:
            with self.crashlog_lock:
                if self.crashlog is not None:
                    logdump = open(self.crashlog, "w")
                else:
                    logdump = NamedTemporaryFile(
                        suffix="_niclassify_crashlog.log",
                        mode="w",
                        encoding="utf8",
                        delete=False,
                    )

                message = " ".join(
                    [
                        "NIClassify encountered an error and the program was aborted.",
                        f"A complete debug-level log has been saved at {logdump.name}",
                    ]
                )
                if self.crashlog is None:
                    atexit.register(lambda: print(message))
                self.crashlog = logdump.name
                # filter out the 'end log' to ensure it only appears at the end
                self.logbuffer = [log for log in self.logbuffer if log != message]
                self.logbuffer.append(message)
                for log in self.logbuffer:
                    # strip rich markup
                    logdump.write(re.sub(r"(?<!\\)\[[^\]]+\]", "", log))
                    logdump.write("\n")
                logdump.close()
                raise typer.Exit(code=1)

    def confirm(self, *message: str, abort=False, allow_pre_confirm=True):
        """Get a simply yes/no response from the user."""

        if allow_pre_confirm and self.pre_confirm:
            self.log(f"[italic bright_black]{' '.join(message)}: y[/]")
            return True

        return typer.confirm(" ".join(message), abort=abort)

    def abort(self) -> None:
        raise typer.Abort()

    # TODO make select and select_multiple accept zero-length responses and handle them appropriately

    def select(
        self,
        prompt: str,
        options: List[str],
        allow_empty: bool = False,
        abort: bool = False,
    ) -> Union[str, None]:
        self.debug("Options:")
        self.debug("\n".join([f"{i + 1}) {v}" for i, v in enumerate(options)]))

        print(columnize(options, dry_run=True, number=True))
        index = 0
        while index is not None and (index < 1 or index > len(options) + 1):
            response = typer.prompt(
                text=f"{prompt} (number)", type=str, default="", show_default=False
            )
            if len(response) == 0 and not (allow_empty or abort):
                continue
            if re.search("[^0-9]", response) is not None:
                continue
            index = int(response) if len(response) > 0 else None
        selection = options[index - 1] if index is not None else None

        self.debug(f"User selection: {selection}")
        if selection is None and abort:
            raise typer.Abort()
        return selection

    def select_multiple(
        self,
        prompt: str,
        options: List[str],
        abort: bool = False,
        allow_empty: bool = False,
    ) -> Union[str, None]:
        self.debug("/n".join(options))

        print(columnize(options, dry_run=True, number=True))
        indexes = [0]
        while indexes is not None and any(
            (i < 1 or i > len(options) + 1 for i in indexes)
        ):
            response = typer.prompt(
                text=f"{prompt} (numbers separated by comma)",
                type=str,
                default="",
                show_default=False,
            )
            if len(response) == 0 and not (allow_empty or abort):
                continue
            if re.search("[^0-9, ]", response) is not None:
                continue
            if len(response) == 0:
                indexes = None
                continue
            indexes = list({int(i) for i in response.replace(" ", "").split(",")})
        selection = [options[i - 1] for i in indexes] if indexes is not None else []

        self.debug("User selection:")
        self.debug("\n".join(selection))
        if len(selection) == 0 and abort:
            raise typer.Abort()
        return selection

    def confirm_overwrite(self, file: Path, abort=False) -> bool:
        if file.exists():
            return self.confirm(
                f"File {file.absolute()} already exists. Overwrite?", abort=abort
            )
        file.parent.mkdir(exist_ok=True, parents=True)
        return True

    def confirm_multiple_overwrite(self, files: List[Path], abort=False) -> bool:
        overwrite_count = len([True for file in files if file.exists()])
        if overwrite_count > 0:
            return self.confirm(
                f"{overwrite_count} files will be overwritten. Continue?", abort=abort
            )
        for file in files:
            file.parent.mkdir(exist_ok=True, parents=True)
        return True

    @contextmanager
    def spin(self, transient=False):
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

    @contextmanager
    def progress(self, transient=False, percent=False):
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
                transient=transient,
            ) as progress:
                CONTEXT["context"] = progress
                yield progress
        finally:
            CONTEXT["context"] = None
