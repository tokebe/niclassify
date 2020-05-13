try:
    import os
    # import sys
    import argparse
    import re
    import xlrd

    import pandas as pd
    import matplotlib as plt
    import seaborn as sns
    import numpy as np
    import json

    from subprocess import call

    from sklearn import preprocessing

    from sklearn.impute import SimpleImputer
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold

    from sklearn.tree import export_graphviz

    # from itertools import chain, combinations
except ModuleNotFoundError:
    print("Missing required modules. Install requirements by running")
    print("'python -m pip install -r requirements.txt'")


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
        "dcol",
        help="column number (1-indexed)\
             range of clustering data in format #:#")
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
        "--predicton",
        nargs='?',
        default=None,
        help="Use a previously saved random forest to predict on data, \
            specified by given path/filename")

    # parse and get arguments
    return parser.parse_args()
