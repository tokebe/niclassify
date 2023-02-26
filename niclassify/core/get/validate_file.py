from pathlib import Path
from ..utils import read_data
import json
import dask.dataframe as dd
from ..interfaces import Handler
from pandas.errors import EmptyDataError, ParserError
from multiprocessing import cpu_count

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.json") as nansfile:
    NANS = json.load(nansfile)


def validate_file(file: Path, handler: Handler, cores: int = cpu_count()) -> None:
    with handler.spin() as spinner:
        task = spinner.add_task(description="Validating file...", total=1)
        try:
            data = read_data(file)
            retrieved_count = data.shape[0].compute(num_workers=cores)
        except ParserError:
            handler.error(handler.prefab.BOLD_FILE_ERR, abort=True)
        except EmptyDataError:
            handler.error(handler.prefab.BOLD_NO_OBSERVATIONS, abort=True)
        except UnicodeDecodeError:
            handler.error(handler.prefab.RESPONSE_DECODE_ERR, abort=True)
        spinner.update(task, description="Validating file...done.", completed=1)
    handler.log(f"Successfully retrieved {retrieved_count} samples.")
