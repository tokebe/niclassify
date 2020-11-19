"""
General file and other interaction utilities.

Generally you want to import by importing the directory, utilities, and
accessing by utilities.function (instead of utilities.general_utils.function).
"""

import json
import os
import shutil
import subprocess
import sys
import xlrd

import pandas as pd

import importlib.resources as pkg_resources
from . import config


# possible null values to be converted to np.nan
with pkg_resources.open_text(config, "nans.json") as nansfile:
    NANS = json.load(nansfile)["nans"]

MAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../"
)

required_folders = [
    os.path.join(MAIN_PATH, "output/"),
    os.path.join(MAIN_PATH, "output/classifiers/"),
    os.path.join(MAIN_PATH, "output/logs/"),
    os.path.join(MAIN_PATH, "output/logs/delim"),
    os.path.join(MAIN_PATH, "output/logs/delim/tree"),
    os.path.join(MAIN_PATH, "output/logs/delim/delim"),
    os.path.join(MAIN_PATH, "output/logs/ftgen/")
]


def assure_path():
    """
    Assure that all required folders exist.

    Creates required folders if they do not.
    """
    for f in required_folders:
        if not os.path.exists(f):
            os.makedirs(f)


def clean_folder(path):
    """Delete contents of a folder.
    If supplied file, deletes the file.
    The folder is removed and replaced.

    Args:
        path (str): folder pathname.
    """
    path = os.path.realpath(path)
    if os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def get_data(filename, excel_sheet=None):
    """
    Get raw data from a given filename.

    Args:
        parser (ArgumentParser): The argument parse for the program. Used for
            raising errors.
        filename (str): The file path/name.
        excel_sheet (str): The sheet name if using an excel_sheet sheet.
            Defaults to None.

    Returns:
        DataFrame: The extracted DataFrame.

    """
    # check if filename is string
    if type(filename) is not str:
        raise TypeError("filename is not string")

    # convert filename to proper path
    if not os.path.isabs(filename):
        filename = os.path.join(MAIN_PATH, "data/" + filename)

    # check if filename exists
    if not os.path.exists(filename):
        raise ValueError("file {} does not exist.".format(filename))

    # get raw data
    if (os.path.splitext(filename)[1]
        in [".xlsx", ".xlsm", ".xlsb",
            ".xltx", ".xltm", ".xls",
            ".xlt", ".xml"
            ]):  # using excel_sheet file
        if excel_sheet is not None:  # sheet given
            if excel_sheet.isdigit():  # sheet number
                raw_data = pd.read_excel(
                    filename,
                    sheet_name=int(excel_sheet) - 1,
                    na_values=NANS,
                    keep_default_na=True,
                    # engine="python"
                )
            else:  # sheet name
                raw_data = pd.read_excel(
                    filename,
                    sheet_name=excel_sheet,
                    na_values=NANS,
                    keep_default_na=True,
                    # engine="python"
                )
        else:  # sheet not given; use default first sheet
            raw_data = pd.read_excel(
                filename,
                sheet_name=0,
                na_values=NANS,
                keep_default_na=True,
                # engine="python"
            )

    elif ".csv" in os.path.splitext(filename)[1]:  # using csv
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True,
            engine="python"
        )

    elif ".tsv" in os.path.splitext(filename)[1]:  # using tsv
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True,
            sep="\t",
            engine="python"
        )

    # using txt; must figure out delimiter
    elif ".txt" in os.path.splitext(filename)[1]:
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True,
            sep=None,
            engine="python"
        )
    else:  # invalid extension
        raise TypeError(
            "data file type is unsupported, or file extension not included")

    return raw_data  # return extracted data


def keyboardInterruptHandler(signal, frame):
    """Handle keyboard interrupts.

    Likely to go unused, but good to have.
    """
    exit(0)


def view_open_file(filename):
    """
    Open a file or directory with system default.

    Args:
        filename (str): Path to file, should be absolute.
    """
    filename = os.path.realpath(filename)
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
