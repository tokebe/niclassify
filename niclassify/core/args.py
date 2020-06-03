"""Setup for commend-line arguments on run-time.

Change to add new arguments are required.
"""
try:
    import argparse
    import logging
    import os
    import re
    import signal

    import pandas as pd

    from PyInquirer import prompt, Validator, ValidationError

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

from .utilities import *

print("args: {}".format(MAIN_PATH))

SUPPORTED_TYPES = [
    ".csv",
    ".xlsx",
    ".tsv",
    ".txt"
]


def keyboardInterruptHandler(signal, frame):
    exit(0)


def getargs():
    """Create argument parser and parse arguments.

    Returns:
        tuple: Argument parser and parsed argument NameSpace.

    """
    # set up multiple subparsers for program modes
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help='program modes',
        dest="mode")

    # set up subparser for training mode
    train_parser = subparsers.add_parser(
        "train",
        help="train a new classifier")
    train_parser.add_argument(
        "data",
        help="path/filename of data to be imported for classification")
    train_parser.add_argument(
        "data_cols",
        help="column number (1-indexed)\
             range of clustering data in format #:#")
    train_parser.add_argument(
        "class_column",
        help="column name for classification label to be predicted")
    train_parser.add_argument(
        "-e",
        "--excel_sheet",
        nargs='?',
        default=1,
        help="sheet number (1-indexed) or name, if importing from excel.\
        Defaults to first sheet if not provided.")
    train_parser.add_argument(
        "-o",
        "--out",
        nargs='?',
        default="output/output.csv",
        help="path/filename of output file, defaults to output/output.csv. \
            Only export to csv is supported.")
    train_parser.add_argument(
        "-n",
        "--nanval",
        nargs='*',
        default=None,
        help="additional values to be counted as NA/NaN")
    train_parser.add_argument(
        "-m",
        "--multirun",
        nargs='?',
        default=1,
        type=int,
        help="Train m number of classifiers and \
            select the best-performing one")

    # set up a subparser for prediction mode
    predict_parser = subparsers.add_parser(
        "predict",
        help="use an existing classifier")
    predict_parser.add_argument(
        "data",
        help="path/filename of data to be imported for classification")
    predict_parser.add_argument(
        "feature_cols",
        help="column number (1-indexed)\
             range of feature data in format #:#")
    predict_parser.add_argument(
        "predict_using",
        help="Use a previously saved random forest to predict on data, \
            specified by given path/filename")
    predict_parser.add_argument(
        "-e",
        "--excel_sheet",
        nargs='?',
        default=1,
        help="sheet number (1-indexed) or name, if importing from excel.\
        Defaults to first sheet if not provided.")
    predict_parser.add_argument(
        "-o",
        "--out",
        nargs='?',
        default="output.csv",
        help="path/filename of output file, defaults to output.csv. \
            Only export to csv is supported. If an absolute path is given \
            output will be placed there instead of internal output folder.")
    predict_parser.add_argument(
        "-n",
        "--nanval",
        nargs='*',
        help="additional values to be counted as NA/NaN")

    # set up a parser for using the interactive mode
    subparsers.add_parser(
        "interactive",
        help="use program in interactive mode, instead of using\
        command-line arguments")

    # parse and get arguments
    return parser, parser.parse_args()


class FileValidator(Validator):
    """Class for validating files."""

    def validate(self, document):
        """Validate a given file path.

        Args:
            document (unkown): whatever prompt hands off for validation.

        Raises:
            ValidationError: In the event that file path doesn not exist or is
                unsupported type.

        """
        ok = True
        ok = os.path.exists(document.text)
        extension = "." + document.text.split(".")[-1]
        ok = extension in SUPPORTED_TYPES
        if not ok:
            raise ValidationError(
                message="File type {} not supported.".format(extension),
                cursor_position=len(document.text))


def interactive_mode():
    """Get all required arguments in a (more) user-friendly format.

    Returns:
        tuple: all required arguments.

    """
    # set up ctrl-c exit
    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    # placeholders and default values
    data_file = None
    selected_cols = None
    class_column = None
    multirun = 1
    classifier_file = None
    output_filename = None
    nans = None

    data_files = os.listdir(
        os.path.join(
            MAIN_PATH,
            "data/"
        )
    )
    data_files.append("OTHER (provide path)")

    # get data file path
    questions = [{
        "type": "list",
        "name": "file",
        "message": "please choose a data file:",
        "choices": data_files
    }]

    data_file = prompt(questions)["file"]

    if data_file == "OTHER (provide path)":
        questions = [{
            "type": "input",
            "name": "path",
            "message": "please provide the path to the file:",
            "validate": FileValidator
        }]
        data_file = prompt(questions)["path"]
    else:
        data_file = os.path.join(MAIN_PATH, "data/", data_file)

    # handle excel sheets
    if data_file.split(".")[-1] == "xlsx":
        print("excel sheet detected.")

        all_sheets = pd.read_excel(data_file, None)

        if len(all_sheets) == 1:
            print("first (only) sheet automatically selected.")
            excel_sheet = 1
        else:
            questions = [{
                "type": "list",
                "name": "selection_mode",
                "message": "would you like to select a sheet from a list or enter a sheet name?",
                "choices": ["select from list", "enter name"]
            }]

            if prompt(questions)["selection_mode"] == "select from list":
                questions = [{
                    "type": "list",
                    "name": "excel_sheet",
                    "message": "select a sheet from the list:",
                    "choices": all_sheets.keys()
                }]

            else:
                questions = [{
                    "type": "input",
                    "name": "excel_sheet",
                    "message": "input a sheet name:",
                    "validate": lambda val: val in all_sheets.keys()
                }]
            excel_sheet = prompt(questions)[excel_sheet]
    else:
        excel_sheet = None

    # get columns for data
    questions = [{
        "type": "list",
        "name": "selection_mode",
        "message": "would you like to select data columns by name or provide a range?",
        "choices": ["select by name", "input range"]
    }]

    # open file to get column names
    columns = get_data(
        data_file, excel_sheet).columns.values.tolist()

    if prompt(questions)["selection_mode"] == "select by name":
        questions = [{
            "type": "checkbox",
            "name": "column_select",
            "message": "please select columns which contain data to train on:",
            "choices": [{'name': col} for col in columns]
        }]
        selected_cols = prompt(questions)["column_select"]
    else:
        questions = [{
            "type": "input",
            "name": "column_select",
            "message": "please select columns which contain data to train on:",
            "validate": (lambda val:
                         re.match("[0-9]+:[0-9]+", val) and
                         ([int(x) for x in val.split(":")][0] in
                          range(1, len(columns) + 1) and
                          [int(x) for x in val.split(":")][1] in
                          range(1, len(columns) + 1)))
            # this lambda is hideous and evil and for the time being a simple
            # way to solve the problem. I need access to the value, and to the
            # columns variable.
        }]
        sel = [int(i) for i in prompt(questions)["column_select"].split(":")]
        selected_cols = columns[sel[0]-1:sel[1]+1]

    questions = [{
        "type": "list",
        "name": "useage_mode",
        "message": "would you like to train a new model or apply a trained model?",
        "choices": ["train new model", "predict using trained model"]
    }]

    if prompt(questions)["useage_mode"] == "train new model":
        mode = "train"
        questions = [{
            "type": "list",
            "name": "select_mode",
            "message": "would you like to select the class label column from a list or enter its name?",
            "choices": ["select from list", "input column name"]
        }]

        if prompt(questions)["select_mode"] == "select from list":
            questions = [{
                "type": "list",
                "name": "column_name",
                "message": "please select the column containing known class labels:",
                "choices": [col for col in columns if col not in selected_cols]
            }]
        else:
            questions = [{
                "type": "input",
                "name": "column_name",
                "message": "please input the column containing known class labels:",
                "validate": (lambda val:
                             val in [col for col in
                                     columns if
                                     col not in selected_cols])
            }]
        class_column = prompt(questions)["column_name"]
        questions = [{
            "type": "list",
            "name": "multirun",
            "message": "how many classifiers would you like to generate (the best will be chosen)",
            "choices": ["1", "10", "50", "100", "500", "1000"]
        }]
        multirun = int(prompt(questions)["multirun"])

    else:
        mode = "predict"
        classifier_files = os.listdir(
            os.path.join(MAIN_PATH, "output/classifiers")
        )
        classifier_files.append("OTHER (provide path)")

        # get data file path
        questions = [{
            "type": "list",
            "name": "file",
            "message": "please choose a classifier file:",
            "choices": classifier_files
        }]

        classifier_file = prompt(questions)["file"]

        if classifier_file == "OTHER (provide path)":

            questions = [{
                "type": "input",
                "name": "path",
                "message": "please provide the path to the file:",
                "validate": lambda val: os.path.exists(val)
            }]

            classifier_file = prompt(questions)["path"]
        else:
            classifier_file = os.path.join(
                MAIN_PATH, "output/classifiers/", classifier_file)

    questions = [{
        "type": "input",
        "name": "path",
        "message": "please provide a name for output files: (without file extension)"
    }]
    output_filename = (prompt(questions)["path"] + ".csv")
    print("ARGS: GOT OUTPUT PROMPT RESPONSE: {}".format(output_filename))
    print("ARGS: MAIN_PATH: {}".format(MAIN_PATH))
    output_filename = os.path.join(MAIN_PATH, "output/" + output_filename)
    print("args: passing output filename of {}".format(output_filename))

    for n in NANS:
        print("'" + n + "'")

    questions = [{
        "type": "list",
        "name": "nans",
        "message": "are there any N/A value types in the data which are not present in the list above?",
        "choices": ["no", "yes"]
    }]

    if prompt(questions)["nans"] == "yes":
        questions = [{
            "type": "input",
            "name": "nans",
            "message": "please input any other N/A formats, separated by a space:"
        }]
        nans = prompt(questions)["nans"].split(" ")

    return (mode,
            data_file,
            excel_sheet,
            selected_cols,
            class_column,
            multirun,
            classifier_file,
            output_filename,
            nans)


if __name__ == "__main__":
    interactive_mode()
