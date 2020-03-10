import argparse
import re
import pandas as pd

# setup arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "data",
    help="path/filename of data to be imported for classification")
parser.add_argument(
    "dcol",
    help="column number range of clustering data in format #:#")
parser.add_argument(
    "clabel",
    help="column name for classification label to be predicted")
parser.add_argument(
    "-e",
    "--excel",
    nargs='?',
    default=1,
    help="sheet number (1-indexed) or name, if importing from excel.\
    Defaults to first sheet if not provided.")
parser.add_argument(
    "-o",
    "--out",
    nargs='?',
    default="output/output.csv",
    help="path/filename of output file, defaults to output/output.csv. \
        Only export to csv is supported.")
parser.add_argument(
    "-w",
    "--weights",
    nargs='*',
    help="column name(s) of variables to use in weighting clustering")

# parse and get arguments
args = parser.parse_args()

# get raw data
if ".xlsx" in args.data[-5:]:
    if args.excel is not None:
        if args.excel.isdigit():
            raw_data = pd.read_csv(args.data, sheet_name=int(args.excel) - 1)
        else:
            raw_data = pd.read_csv(args.data, sheet_name=args.excel)
elif ".csv" in args.data[-4:]:
    raw_data = pd.read_csv(args.data)
else:
    parser.error(
        "data file type is unsupported, or file extension not included")

# split data into metadata and feature data
if not re.match("[0-9]:[0-9]", args.dcol):
    parser.error("data column selection range format invalid (see -h).")
else:
    dcol = args.dcol.split(":")

data = raw_data.iloc[:, dcol[0]:dcol[1]]
metadata = raw_data.drop(raw_data.columns[dcol[0]:dcol[1]], axis=1)


print(args)

"""Generate a new datasheet with predictions.

    runtime args:
        python kmeans-auto.py a b c
        a: data filename
        b: first column number of features (1-indexed)
        c: output filename (optional)

    Args:
        args (str): list of runtime arguments

    """
