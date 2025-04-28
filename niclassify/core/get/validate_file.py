from pathlib import Path
from typing import cast
from niclassify.core.utils import read_data
import yaml
from niclassify.core.interfaces import Handler
from pandas.errors import EmptyDataError, ParserError
import polars as pl

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.yaml") as nansfile:
    NANS = yaml.safe_load(nansfile)


def validate_file(file: Path, handler: Handler) -> None:
    with handler.spin() as spinner:
        task = spinner.add_task(description="Validating file...", total=1)
        try:
            data = read_data(file)
            retrieved_count = cast(
                int, data.select(pl.len()).collect(streaming=True).item()
            )
        except pl.NoDataError:
            handler.error(handler.prefab.ERR_BOLD_NO_OBSERVATIONS, abort=True)
            exit(1)
        except pl.PolarsError:
            handler.error(handler.prefab.ERR_BOLD_FILE, abort=True)
            exit(1)
        except UnicodeDecodeError:
            handler.error(handler.prefab.ERR_RESPONSE_DECODE, abort=True)
            exit(1)
        spinner.update(task, description="Validating file...done.", completed=1)
    handler.log(f"Successfully retrieved {retrieved_count} samples.")
