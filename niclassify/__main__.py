# from PyInquirer import prompt, print_json, Separator
import typer
from rich import print
from typing import List, Optional
from pathlib import Path
from enum import Enum
from typer.core import TyperGroup

from niclassify.cli import (
    _get,
    _identify,
    _lookup,
    _filter,
    _align,
    _delimit,
    _featgen,
    _train,
    _predict,
    _column_select,
)

# Order commands in order defined
class NaturalOrderGroup(TyperGroup):
    def list_commands(self, ctx):
        return list(self.commands)


app = typer.Typer(rich_markup_mode="rich", cls=NaturalOrderGroup)
app.command(name="get")(_get)
app.command(name="identify")(_identify)
app.command(name="lookup")(_lookup)
app.command(name="filter")(_filter)
app.command(name="align")(_align)
app.command(name="delimit")(_delimit)
app.command(name="featgen")(_featgen)
app.command(name="select")(_column_select)
app.command(name="train")(_train)
app.command(name="predict")(_predict)

# TODO get rid of globals, just pass in though map_partition and apply args

# TODO refactor confirm_overwrite to be a method of handler

# TODO add "leave blank to use system file browser"

# TODO for the lead-up to training, try to add some sort of weight so you don't have to worry about data duplication
# basically, minimize how much data needs to be loaded to memory for training purposes.

# TODO figure out what to call the composition of all data prep command?

# TODO debug logging

# TODO add argument to prompt instead of defaults when using interactive mode
# maybe just --interactive?

# TODO confirm overwrite if file exists, and -y argument to bypass this

# TODO add core count per-command and global to overwrite (check that interaction)

# TODO add project options

# TODO write full niclassify composition

# TODO in composed interactive mode, ask if user would prefer to input file or use file browser (if there's a module to support that)

# TODO type annotate everything that seems reasonable to

# TODO move away from handler prefabs?


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    # TODO use environment variables to set arguments when composing commands
    if ctx.invoked_subcommand is not None:
        return
    print("default")


if __name__ == "__main__":
    app()


# LATERER IDEAS
# - some sort of automated script for retrieving all sequences from same species/identified species?
#   essentially something to 'bolster' existing sequences...not sure if totally useful
