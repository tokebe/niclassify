"""
A set of utilties for handling the creation and application of a classifier.

Strictly to be used in concert with the rest of the niclassify package.
"""
try:
    import json
    import logging
    import os
    import re
    import subprocess
    import sys
    import xlrd

    import matplotlib as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer

    import importlib.resources as pkg_resources


except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

from . import config
from .classifiers import AutoClassifier

sns.set()

# possible null values to be converted to np.nan
with pkg_resources.open_text(config, "nans.json") as nansfile:
    NANS = json.load(nansfile)["nans"]

MAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../"
)

required_folders = [
    os.path.join(MAIN_PATH, "output/"),
    os.path.join(MAIN_PATH, "output/classifiers/"),
    os.path.join(MAIN_PATH, "output/logs/"),
]


def assure_path():
    """
    Assure that all required folders exist.

    Creates required folders if they do not.
    """
    for f in required_folders:
        if not os.path.exists(f):
            os.makedirs(f)


def get_col_range(rangestring):
    """
    Get column range from a given string.

    Args:
        parser (ArgumentParser): The argument parse for the program. Used for
            raising errors.
        rangestring (str): A string in format ##:##.

    Returns:
        list: a list containing the lower and upper values of the range.

    """
    # get feature data column range
    if not re.match("[0-9]+:[0-9]+", rangestring):
        logging.error("data column selection range format invalid (see -h).")
        raise ValueError("data column selection range format invalid")
    else:
        feature_cols = rangestring.split(":")
        feature_cols = [int(x) for x in feature_cols]
        # adjust numbers to be 0-indexed and work for slice
        feature_cols[0] -= 1
        feature_cols[1] += 1

    return feature_cols


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
    if ".xlsx" in filename[-5:]:  # using excel_sheet file
        if excel_sheet is not None:  # sheet given
            if excel_sheet.isdigit():  # sheet number
                raw_data = pd.read_excel(
                    filename,
                    sheet_name=int(excel_sheet) - 1,
                    na_values=NANS,
                    keep_default_na=True)
            else:  # sheet name
                raw_data = pd.read_excel(
                    filename,
                    sheet_name=excel_sheet,
                    na_values=NANS,
                    keep_default_na=True)
        else:  # sheet not given; use default first sheet
            raw_data = pd.read_excel(
                filename,
                sheet_name=0,
                na_values=NANS,
                keep_default_na=True)

    elif ".csv" in filename[-4:]:  # using csv
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True)
    elif ".tsv" in filename[-4:]:  # using tsv
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True,
            sep="\t")
    elif ".txt" in filename[-4:]:  # using txt; must figure out deliniation
        raw_data = pd.read_csv(
            filename,
            na_values=NANS,
            keep_default_na=True,
            sep=None,
            engine="python")
    else:  # invalid extension
        logging.error(
            "data file type is unsupported, or file extension not included")
        exit(-1)

    # replace unknown values with nan (should be redundant)
    raw_data.replace({
        "unknown": np.nan,
        "Unknown": np.nan,
        "0": np.nan})  # replace string zero with nan (shouldn't exist)

    return raw_data  # return extracted data


def get_known(data, metadata, class_column):
    """
    Get only known data/metadata given the class label.

    Args:
        data (DataFrame): All data, preferably normalized.
        metadata (DataFrame): All metadata, including class label.
        class_column (str): Name of class label column in metadata.

    Returns:
        tuple: tuple of known data and metadata DataFrames, where class label
            is already known.

    """
    # split into known and unknown
    logging.info("splitting data...")
    data_known = data[metadata[class_column].notnull()]
    metadata_known = metadata[metadata[class_column].notnull()]
    # reset indices
    data_known.reset_index(drop=True, inplace=True)
    metadata_known.reset_index(drop=True, inplace=True)

    # remove null rows from known data
    logging.info("extracting fully known data...")
    data_known = data_known.dropna()
    metadata_known = metadata_known.iloc[data_known.index]
    data_known.reset_index(drop=True, inplace=True)
    metadata_known.reset_index(drop=True, inplace=True)

    return data_known, metadata_known


def impute_data(data):
    """
    Impute the given data.

    Args:
        data (DataFrame): Data. Preferably normalized.

    Returns:
        DataFrame: The imputed data.

    """
    # get categorical columns for dummy variable encoding
    category_cols = list(data.select_dtypes(
        exclude=[np.number]).columns.values)
    data_np = pd.get_dummies(
        data,
        columns=category_cols,
        prefix=category_cols, drop_first=True)
    feature_cols = data_np.columns.values
    data_np = data_np.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_np = imp_mean.fit_transform(data_np)
    # return to dataframe
    data = pd.DataFrame(data_np)  # removed a scaler fit-transform here
    data.columns = feature_cols

    return data


def keyboardInterruptHandler(signal, frame):
    """Handle keyboard interrupts.

    Likely to go unused, but good to have.
    """
    exit(0)


def load_classifier(filename):
    """
    Load a saved classifier.

    Args:
        filename (str): Path/name of classifier.

    Returns:
        Classifier: The loaded classifier.

    """
    if not os.path.isabs(filename):
        filename = os.path.join(MAIN_PATH, "output/classifiers/", filename)

    # check if filename exists
    if not os.path.exists(filename):
        raise ValueError("file {} does not exist.".format(filename))

    from joblib import load
    classifier = load(filename)

    if not isinstance(classifier, AutoClassifier):
        raise TypeError("loaded file is not compatible (not AutoClassifier)")

    return classifier


def make_confm(clf, features_known, metadata_known, class_column):
    """
    Make a confusion matrix plot of a trained classifier.

    Args:
        clf (Classifier): A classifier.
        data_known (DataFrame): Known Data.
        metadata_known (DataFrame): Known Metadata, including class labels.
        class_column (str): Name of class label column in metadata.

    Returns:
        fig (figure): A matplotlib figure containing the graph.

    """
    # type error checking
    if type(features_known) is not pd.DataFrame:
        raise TypeError("Cannot save: features_known is not DataFrame.")
    if type(metadata_known) is not pd.DataFrame:
        raise TypeError("Cannot save: metadata_known is not DataFrame.")

    fig, ax = plt.pyplot.subplots(nrows=1, ncols=1)
    metrics.plot_confusion_matrix(
        clf,
        features_known,
        metadata_known[class_column],
        ax=ax,
        normalize="true")
    ax.grid(False)

    return fig


def output_graph(data, predict, outfname):
    """
    Output a pairplot of the predicted values.

    Args:
        data (DataFrame): Data predicted on. Preferably normalized.
        predict (DataFrame): Class label predictions.
        out (str): Output file path/name
    """
    # type error checking
    if type(predict) is not pd.DataFrame and type(predict) is not pd.Series:
        raise TypeError("Cannot save: predict is not DataFrame or Series.")

    # change data in frame to be useable for graph
    df = pd.concat([data, predict], axis=1)
    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data.shape[1]],
        hue="predict",
        diag_kind='hist'
    )
    if not os.path.isabs(outfname):
        outfname = os.path.join(MAIN_PATH, "output/" + outfname)
    out.savefig("{}.png".format(outfname))


def save_clf(clf):
    """Save a given AutoClassifier.

    Args:
        clf (AutoClassifier): A trained classifier.
    """
    from joblib import dump
    i = 0
    while os.path.exists(
        os.path.join(
            MAIN_PATH,
            "output/classifiers/{}{}.gz".format(
                clf.__class__.__name__, i)
        )
    ):
        i += 1
    dump(
        clf,
        os.path.join(
            MAIN_PATH,
            "output/classifiers/{}{}.gz".format(
                clf.__class__.__name__, i)
        )
    )


def save_clf_dialog(clf):
    """
    Present the user with the option to save the given AutoClassifier.

    Args:
        clf (AutoClassifier): A trained AutoClassifier.

    """
    print("would you like to save the trained classifier? (y/n)")
    while 1:
        answer = input("> ")
        if answer in ["y", "yes"]:
            save_clf(clf)
            break
        elif answer in ["n", "no"]:
            break
        else:
            continue


def save_confm(clf, features_known, metadata_known, class_column, out):
    """
    Save a confusion matrix plot of a trained classifier.

    Args:
        clf (AutoClassifier): An AutoClassifier.
        data_known (DataFrame): Known Data.
        metadata_known (DataFrame): Known Metadata, including class labels.
        class_column (str): Name of class label column in metadata.
        out (str): file path/name for output image.
    """
    # type error checking
    if type(features_known) is not pd.DataFrame:
        raise TypeError("Cannot save: features_known is not DataFrame.")
    if type(metadata_known) is not pd.DataFrame:
        raise TypeError("Cannot save: metadata_known is not DataFrame.")

    fig = make_confm(clf.clf, features_known, metadata_known, class_column)
    if not os.path.isabs(out):
        out = os.path.join(MAIN_PATH, "output/" + out)
    fig.savefig("{}.cm.png".format(out))


def save_predictions(metadata, predict, feature_norm, out, predict_prob=None):
    """
    Save a given set of predictions to the given output filename.

    Args:
        metadata (DataFrame): The full set of metadata.
        predict (Series): The predictions given for the feature data.
        feature_norm (DataFrame): Normalized feature data.
        out (str): The output filename.
        predict_prob (DataFrame, optional): The class prediction probabilities.
            Defaults to None.
    """
    # type error checking
    if type(metadata) is not pd.DataFrame:
        raise TypeError("Cannot save: metadata is not DataFrame.")
    if type(feature_norm) is not pd.DataFrame:
        raise TypeError("Cannot save: feature_norm is not DataFrame.")
    if (type(predict) is not pd.DataFrame
            and type(predict) is not pd.Series):
        raise TypeError("Cannot save: predict is not DataFrame or Series.")
    if (type(predict_prob) is not pd.DataFrame
            and type(predict_prob) is not pd.Series):
        raise TypeError(
            "Cannot save: predict_prob is not DataFrame or Series.")

    logging.info("saving new output...")
    if predict_prob is not None:
        df = pd.concat([metadata, predict, predict_prob, feature_norm], axis=1)
    else:
        df = pd.concat([metadata, predict, feature_norm], axis=1)
    try:
        if not os.path.isabs(out):
            out = os.path.join(MAIN_PATH, "output/" + out)

        output_path = os.path.join(
            "/".join(out.replace("\\", "/").split("/")[:-1]))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("  saving files to path: {}".format(
            os.path.realpath(output_path)))
        df.to_csv(out, index=False)
    except (KeyError, FileNotFoundError, OSError):
        logging.error("output folder creation failed.")
        exit(-1)


def scale_data(data):
    """
    Scale given data to be normalized.

    Args:
        data (DataFrame): Data to be scaled.

    Returns:
        Dataframe: Scaled data.

    """
    # scale data
    logging.info("scaling data...")
    # get categorical columns for dummy variable encoding
    category_cols = list(
        data.select_dtypes(exclude=[np.number]).columns.values)
    data_np = pd.get_dummies(
        data,
        columns=category_cols,
        prefix=category_cols, drop_first=True)
    feature_cols = data_np.columns.values  # save column names
    data_np = data_np.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = feature_cols  # add column names back

    return data_norm


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
