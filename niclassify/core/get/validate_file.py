from pathlib import Path
from ..utils import read_data
import json
import dask.dataframe as dd
from ..interfaces import Handler
from pandas.errors import EmptyDataError, ParserError


NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.json") as nansfile:
    NANS = json.load(nansfile)


def validate_file(file: Path, handler: Handler) -> None:
    with handler.spin() as spinner:
        task = spinner.add_task(description="Validating file...", total=1)
        try:
            data = read_data(file)
            retrieved_count = data.shape[0].compute()
            handler.message(f"Successfully retrieved {retrieved_count} samples!")
        except ParserError:
            handler.error(handler.prefab.BOLD_FILE_ERR)
        except EmptyDataError:
            handler.error(handler.prefab.BOLD_NO_OBSERVATIONS)
        except UnicodeDecodeError:
            handler.error(handler.prefab.RESPONSE_DECODE_ERR)
        spinner.update(task, description="Validating file...done.", completed=1)
