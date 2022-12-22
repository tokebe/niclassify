from pathlib import Path
from ...cli import Handler
import pandas as pd
import json
from contextlib import contextmanager

NANS = []

with open(Path(__file__).parent.parent.parent / "config/nans.json") as nansfile:
    NANS = json.load(nansfile)


@contextmanager
def read_data(file: Path, handler: Handler, chunked=True):

    # check if file exists
    if not file.exists():
        handler.error(ValueError("file {} does not exist.".format(file)))

    if not chunked:
        data = pd.read_csv(
            file,
            na_values=NANS,
            keep_default_na=True,
            engine="python",
        )
        # chances are something went wrong if there's only one column
        if data.shape[1] == 1:
            data = pd.read_csv(
                file, na_values=NANS, keep_default_na=True, sep="\t", engine="python"
            )  # it's either actually 1 column or now it'll read correctly

    automatic = True

    with pd.read_csv(
        file,
        na_values=NANS,
        keep_default_na=True,
        engine="python",
        chunksize=1,
    ) as reader:
        for chunk in reader:
            if chunk.shape[1] == 1:
                automatic = False
                break
            else:
                yield chunk

    # TODO fix behavior after here

    return data  # return extracted data
