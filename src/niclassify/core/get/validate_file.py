from pathlib import Path
from typing import cast

import polars as pl
import yaml

from niclassify.core.interfaces.handler import Handler
from niclassify.core.utils.read_data import read_data

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.yaml") as nansfile:
    NANS = yaml.safe_load(nansfile)


def validate_file(file: Path, handler: Handler) -> None:
    with handler.spin() as spinner:
        task = spinner.add_task(description="Validating file...", total=1)
        try:
            data = read_data(file, handler=handler)
            retrieved_count = cast(
                int, data.select(pl.len()).collect(engine="streaming").item()
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
