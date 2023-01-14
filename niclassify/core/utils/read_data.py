from pathlib import Path
import dask.dataframe as dd
import json

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.json") as nansfile:
    NANS = json.load(nansfile)


def read_data(file: Path) -> dd.DataFrame:
    data = dd.read_csv(
        file,
        sample=1000000,
        sample_rows=100,
        assume_missing=True,
        na_values=NANS,
        keep_default_na=True,
        engine="python",
        sep=None,
        dtype="object",
    )

    # if there's one column, it probably read wrong somehow
    # otherwise if there *should* be one column, this should be fine
    if len(data.columns) < 2:
        data = dd.read_csv(
            file,
            sample=1000000,
            sample_rows=100,
            assume_missing=True,
            na_values=NANS,
            keep_default_na=True,
            engine="python",
            sep='\t',
            dtype="object",
        )
    return data
