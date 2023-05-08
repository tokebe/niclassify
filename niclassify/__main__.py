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
    _format,
    _write,
)


# Order commands in order defined
class NaturalOrderGroup(TyperGroup):
    def list_commands(self, ctx):
        return list(self.commands)


sample_group = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    help="Operate on tab-delimited sample data.",
)
sample_group.command(name="format")(_format)
sample_group.command(name="get")(_get)
sample_group.command(name="identify")(_identify)
sample_group.command(name="lookup")(_lookup)
sample_group.command(name="filter")(_filter)

fasta_group = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    help="Create and operate on FASTA files.",
)
fasta_group.command(name="write")(_write)
fasta_group.command(name="align")(_align)
fasta_group.command(name="delimit")(_delimit)

feature_group = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    help="Generate and select features for the classifier.",
)
feature_group.command(name="featgen")(_featgen)
feature_group.command(name="select")(_column_select)

classifier_group = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    help="Train classifiers and make predictions.",
)
classifier_group.command(name="train")(_train)
classifier_group.command(name="predict")(_predict)

app = typer.Typer(rich_markup_mode="rich", cls=NaturalOrderGroup)
app.add_typer(sample_group, name="sample")
app.add_typer(fasta_group, name="fasta")
app.add_typer(feature_group, name="feature")
app.add_typer(classifier_group, name="classifier")

# TODO re-install env, see if you still have double of prompt-toolkit
# this seems to be the current thing throwing errors with jupyter notebooks

# TODO use namedTemporaryFile for all output and then copy to output
# this should avoid any issues with overwriting input

# TODO revamp regions schema -- probably just make it a graph
# Using a list of nodes which each define their direct parents and children
# Due to multi-parent, we need to traverse all parents
# Add "Mediterranean Region"
#   contains/is contained by: Atlantic Ocean, Europe, Asia, Africa
# Add "Middle Asia"
# Add "Palearctic ecozone"
# Add "Southeastern Europe"
# Add "Southwestern Europe"
# Add "India"
# Add "Singapore"
# Add "Southern America"
# Add "Western Asia",
# Add "Tropical Indo-Pacific Region"
# Add "West-Central Tropical Africa"
# Add "Indian Subcontinent"
# Add "Malesia"
# Add "Tropical Indo-Pacific Region"
# South America
# Argentina Distrito Federal
# Argentina Northeast
# Northern South America
# Southern America (WGSRPD:8)
# Siberia
# Papuasia
# Madagascar
# France
# New World Tropics
# West-Central Tropical Africa
# South Tropical Africa
# Northeast Tropical Africa
#

# TODO add a command for combining fasta files (because user might be lazy)

# TODO get rid of globals, just pass in though map_partition and apply args

# TODO add "leave blank to use system file browser"
# only implement this for interactive mode to save yourself sanity

# TODO consider adding further hierarchy to commands?
"""
- niclassify
    - interactive (same as just typing niclassify, runs through everything with user-friendly questions)
    - sample
        - format (takes in tsv, asks questions to conform to supported format, interactive-only)
        - concat (concatenates proper-formatted data files)
        - get
        - identify
        - lookup
        - filter
    - fasta
        - write (just writes out to unaligned fasta)
        - combine (takes multiple fastas and combines into one)
        - align (the whole shebang)
        - align-custom (just a passthrough to muscle)
        - trim (for just trimming reading frames)
    - feature
        - generate
        - select
    - classifier
        - train
        - predict
"""

# TODO for the lead-up to training, try to add some sort of weight so you don't have to worry about data duplication
# basically, minimize how much data needs to be loaded to memory for training purposes.
# also, we're gonna want incremental training if data is larger-than-memory
# calculate number of equal-size data splits, then split up trees to each split so it's even

# TODO figure out what to call the composition of all data prep command?

# TODO more debug logging

# TODO add argument to prompt instead of defaults when using interactive mode
# maybe just --interactive?

# TODO add project options

# TODO write full niclassify composition

# TODO type annotate everything that seems reasonable to

# TODO move away from handler prefabs

# TODO implement an automatic path completion
# see https://typer.tiangolo.com/tutorial/options-autocompletion/

# TODO type annotation pass, remove unnecessary typing imports


@app.callback(invoke_without_command=True)
@app.command()
def interactive(ctx: typer.Context):
    """
    Run NIClassify in interactive mode for ease-of-use.
    """
    # TODO use environment variables to set arguments when composing commands?
    if ctx.invoked_subcommand is not None:
        return
    print("default")


if __name__ == "__main__":
    app()


# LATERER IDEAS
# - some sort of automated script for retrieving all sequences from same species/identified species?
#   essentially something to 'bolster' existing sequences...not sure if totally useful
