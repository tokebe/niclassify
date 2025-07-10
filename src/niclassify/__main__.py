from multiprocessing import cpu_count
from typing import Annotated

import typer

from niclassify.cli.classifier.app import classifier_command
from niclassify.cli.fasta.app import fasta_command
from niclassify.cli.feature.app import feature_command
from niclassify.cli.interactive import cli_interactive
from niclassify.cli.sample.app import sample_command
from niclassify.cli.utils.general import NaturalOrderGroup
from niclassify.core.interfaces.handler import Handler

n_cpus = cpu_count()

app = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
app.add_typer(sample_command, name="sample")
app.add_typer(fasta_command, name="fasta")
app.add_typer(feature_command, name="feature")
app.add_typer(classifier_command, name="classifier")

# TODO use namedTemporaryFile for all output and then copy to output
# this should avoid any issues with overwriting input

# TODO: work out more of the region hierarchy in the new regions.yaml
# (there are a lot of empty nodes)


# TODO add "leave blank to use system file browser"
# only implement this for interactive mode to save yourself sanity

"""
- niclassify
    - interactive (same as just typing niclassify, runs through everything with user-friendly questions)
    - sample
        - format (takes in tsv, asks questions to conform to supported format, interactive-only)
        - concat (concatenates proper-formatted data files)
        - get
        - filter
        - identify
        - lookup
    - fasta
        - write (just writes out to unaligned fasta)
        - combine (takes multiple fastas and combines into one)
        - align (the whole shebang)
        - align-custom (just a passthrough to muscle)
        - trim (for just trimming reading frames)
        - delimit
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


# TODO more debug logging

# TODO add project options

# TODO write full niclassify composition

# TODO: if a message is used in multiple places, move it to prefab, otherwise leave it

# TODO implement an automatic path completion
# see https://typer.tiangolo.com/tutorial/options-autocompletion/

# TODO: make handler something that is initialized on first import and then can be
# be imported instead of passed around


# @app.callback(invoke_without_command=True)
@app.command()
def interactive(
    ctx: typer.Context,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Output debug logs to stdout.",
        ),
    ] = False,
):
    """Run NIClassify in interactive mode for ease-of-use."""
    # TODO use environment variables to set arguments when composing commands?
    if ctx.invoked_subcommand is not None:
        return
    handler = Handler(pre_confirm=False, debug=debug)
    cli_interactive(handler)


if __name__ == "__main__":
    app()


# LATERER IDEAS
# - some sort of automated script for retrieving all sequences from same species/identified species?
#   essentially something to 'bolster' existing sequences...not sure if totally useful
