import atexit
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from types import SimpleNamespace
from typing import NoReturn

import typer
import yaml
from InquirerPy import inquirer
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from niclassify.cli.columnize import columnize

# TODO: automatically handle syntaxwarnings, put them in debug logs

# CONTEXT
CONTEXT: dict[str, Progress | None] = {"context": None}

with open(Path(__file__).parent / "prefab.yaml") as prefab_file:
    prefab = SimpleNamespace(**yaml.safe_load(prefab_file))


class Handler:
    """Interface for interaction and log handling.

    Default interface is CLI.
    """

    prefab = prefab  # Pre-written reusable messages, see niclassify/core/interfaces/prefab.yaml

    def __init__(self, pre_confirm: bool = False, debug: bool = False):
        self.pre_confirm = pre_confirm
        self._debug = debug
        self.log_lock = Lock()
        self.logbuffer = []
        self.crashlog = None
        self.crashlog_lock = Lock()

    def prefix_with_indent(self, *message: str, prefix: str) -> str:
        """Return message with prefix, respecting indent."""
        indent = len(message[0]) - len(message[0].lstrip())
        lstripped = " ".join(
            [part.lstrip() if i == 0 else part for i, part in enumerate(message)]
        )
        return f"{' ' * indent}{prefix} {lstripped}"

    def debug(self, *message: str):
        """Print message only if debugging is enabled."""
        if not self._debug:  # Log to buffer regardless
            self.logbuffer.append(" ".join(message))
            return
        self.log(
            "".join(
                [
                    "[italic bright_black]",
                    f"{self.prefix_with_indent(*message, prefix='DEBUG:')}",
                    "[/]",
                ]
            )
        )

    def log(self, *message: str):
        """Log a message."""
        with self.log_lock:
            if CONTEXT["context"] is not None:
                CONTEXT["context"].console.print(" ".join(message))
            else:
                print(" ".join(message))
            self.logbuffer.append(" ".join(message))

    def message(self, *message: str):
        """Log a message and wait for the user to acknowledge."""
        self.log(*message)
        typer.prompt("Press enter to continue", hide_input=True)

    def warning(self, *message: str):
        """Log a message with a warning prefix to grab user attention."""
        self.log(self.prefix_with_indent(*message, prefix="[bold yellow]WARNING:[/]"))

    def error(self, *error: str | Exception, abort: bool | int = False) -> None:
        """Log a message with an error prefix and exit if required.

        If provided with an Exception, the traceback will be printed as well.
        If abort is True or >0, attempt to exit the program with the given code (or 1 if set to True).
        """
        if isinstance(error[0], Exception):
            self.log(
                self.prefix_with_indent(str(error[0]), prefix="[bold red]ERROR:[/]")
            )
            self.log(traceback.format_exc())
        else:
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
                raise typer.Exit(code=1 if not isinstance(abort, int) else abort)

    def confirm(self, *message: str, abort=False, allow_pre_confirm=True):
        """Get a simply yes/no response from the user."""
        if allow_pre_confirm and self.pre_confirm:
            self.log(f"[italic bright_black]{' '.join(message)}: y[/]")
            return True

        return typer.confirm(" ".join(message), abort=abort)

    def abort(self) -> NoReturn:
        raise typer.Abort()

    def select(self, prompt: str, options: list[str], abort: bool = False) -> str:
        """Prompt user to select one item from a list."""
        self.debug("Options:")
        self.debug("\n".join(options))

        selection = inquirer.fuzzy(  # pyright:ignore[reportPrivateImportUsage] InquirerPy is a bit weird
            message=prompt,
            choices=options,
            instruction="(Type to filter, Enter to confirm)",
            info=True,
        ).execute()

        if selection is None and abort:
            raise typer.Abort()

        self.debug(f"User selection: {selection}")
        return selection

    def select_multiple(
        self,
        prompt: str,
        options: list[str],
        allow_empty: bool = False,
        abort: bool = False,
    ) -> list[str] | None:
        """Prompt user to select multiple items from a list."""
        self.debug("Options:")
        self.debug("/n".join(options))

        selection = inquirer.fuzzy(
            message=prompt,
            choices=options,
            multiselect=True,
            instruction="(Type to filter, Tab to select, Enter to confirm)",
            info=True,
            validate=lambda result: len(result) > 0 if not allow_empty else True,
        ).execute()

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

    def confirm_multiple_overwrite(self, files: list[Path], abort=False) -> bool:
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
        """Start one or more spinners in a context manager."""
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
