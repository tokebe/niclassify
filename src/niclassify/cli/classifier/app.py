import typer

from niclassify.cli.classifier.predict import cli_predict
from niclassify.cli.classifier.train import cli_train
from niclassify.cli.utils.general import NaturalOrderGroup

classifier_command = typer.Typer(
    rich_markup_mode="rich",
    cls=NaturalOrderGroup,
    no_args_is_help=True,
    help="Train classifiers and make predictions.",
)
train = classifier_command.command(name="train")(cli_train)
predict = classifier_command.command(name="predict")(cli_predict)
