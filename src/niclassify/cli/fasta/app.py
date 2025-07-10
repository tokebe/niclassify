import typer

from niclassify.cli.fasta.align import cli_align
from niclassify.cli.fasta.delimit import cli_delimit
from niclassify.cli.fasta.trim import cli_trim
from niclassify.cli.fasta.write import cli_write
from niclassify.cli.utils.general import NaturalOrderGroup

fasta_command = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    no_args_is_help=True,
    help="Create and operate on FASTA files.",
)
write = fasta_command.command(name="write")(cli_write)
align = fasta_command.command(name="align")(cli_align)
trim = fasta_command.command(name="trim")(cli_trim)
delimit = fasta_command.command(name="delimit")(cli_delimit)
