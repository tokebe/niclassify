"""Setup for commend-line arguments on run-time.

Change to add new arguments are required.
"""
try:
    import argparse
    import logging

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")


def getargs(parser):
    """Get arguments for argparse.

    Args:
        parser (ArgumentParser): From argparse.

    Returns:
        NameSpace: Argument parser.

    """
    parser.add_argument(
        "data",
        help="path/filename of data to be imported for classification")
    parser.add_argument(
        "data_cols",
        help="column number (1-indexed)\
             range of clustering data in format #:#")
    parser.add_argument(
        "class_column",
        help="column name for classification label to be predicted")
    parser.add_argument(
        "-e",
        "--excel_sheet",
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
        "-n",
        "--nanval",
        nargs='*',
        help="additional values to be counted as NA/NaN")
    parser.add_argument(
        "-m",
        "--multirun",
        nargs='?',
        default=1,
        type=int,
        help="Train m number of classifiers and \
            select the best-performing one")
    parser.add_argument(
        "-p",
        "--predict_using",
        nargs='?',
        default=None,
        help="Use a previously saved random forest to predict on data, \
            specified by given path/filename")

    # parse and get arguments
    return parser.parse_args()
