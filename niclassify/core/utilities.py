"""A set of utilties for handling the creation and application of a classifier.

Strictly to be used in concert with the rest of the niclassify package.
"""
try:
    import os
    import re
    import json
    import xlrd
    import logging

    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import seaborn as sns

    from sklearn import preprocessing
    from sklearn import metrics
    from sklearn.impute import SimpleImputer

    import importlib.resources as pkg_resources


except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

from . import config

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

print("utilties: {}".format(MAIN_PATH))


def assure_path():
    """Assure that all required folders exist.

    Creates required folders if they do not.
    """
    for f in required_folders:
        if not os.path.exists(f):
            os.makedirs(f)


def get_data(filename, excel_sheet=None):
    """Get raw data from a given filename.

    Args:
        parser (ArgumentParser): The argument parse for the program. Used for
            raising errors.
        filename (str): The file path/name.
        excel_sheet (str): The sheet name if using an excel_sheet sheet.
            Defaults to None.

    Returns:
        DataFrame: The extracted DataFrame.

    """
    assert type(filename) is str, "filename must be string"
    assert os.path.exists(
        filename), "file {} does not exist.".format(filename)

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


def get_col_range(rangestring):
    """Get column range from a given string.

    Args:
        parser (ArgumentParser): The argument parse for the program. Used for
            raising errors.
        rangestring (str): A string in format ##:##.

    Returns:
        list: a list containing the lower and upper values of the range.

    """
    assert type(rangestring) is str, "given range format incompatible"

    # get feature data column range
    if not re.match("[0-9]+:[0-9]+", rangestring):
        logging.error("data column selection range format invalid (see -h).")
        exit(-1)
    else:
        data_cols = rangestring.split(":")
        data_cols = [int(x) for x in data_cols]
        data_cols[0] -= 1  # adjust numbers to be 0-indexed and work for slice
        data_cols[1] += 1

    return data_cols


def scale_data(data):
    """Scale given data to be normalized.

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
    data_cols = data_np.columns.values  # save column names
    data_np = data_np.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = data_cols  # add column names back

    return data_norm


def get_known(data, metadata, class_column):
    """Get only known data/metadata given the class label.

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

    # debug. logging.infos counts of each existing class label.
    # v = metadata_known[args.class_column].value_counts()
    # logging.info("  {} {}\n  {} {}".format(
    #     v[0],
    #     v.index[0],
    #     v[1],
    #     v.index[1]))

    return data_known, metadata_known


def make_confm(clf, data_known, metadata_known, class_column):
    """Make a confusion matrix plot of a trained classifier.

    Args:
        clf (Classifier): A classifier.
        data_known (DataFrame): Known Data.
        metadata_known (DataFrame): Known Metadata, including class labels.
        class_column (str): Name of class label column in metadata.

    Returns:
        fig (figure): A matplotlib figure containing the graph.
    """
    fig, ax = plt.pyplot.subplots(nrows=1, ncols=1)
    metrics.plot_confusion_matrix(
        clf,
        data_known,
        metadata_known[class_column],
        ax=ax,
        normalize="true")
    ax.grid(False)

    return fig


def save_confm(clf, data_known, metadata_known, class_column, out):
    """Save a confusion matrix plot of a trained classifier.

    Args:
        clf (Classifier): A classifier.
        data_known (DataFrame): Known Data.
        metadata_known (DataFrame): Known Metadata, including class labels.
        class_column (str): Name of class label column in metadata.
        out (str): file path/name for output image.
    """
    fig = make_confm(clf, data_known, metadata_known, class_column)
    fig.savefig("{}.cm.png".format(out))


def save_predictions(metadata, predict, feature_norm, out, predict_prob=None):
    """Save a given set of predictions to the given output filename.

    Args:
        metadata (DataFrame): The full set of metadata.
        predict (Series): The predictions given for the feature data.
        feature_norm (DataFrame): Normalized feature data.
        out (str): The output filename.
        predict_prob (DataFrame, optional): The class prediction probabilities.
            Defaults to None.
    """
    logging.info("saving new output...")
    if predict_prob is not None:
        df = pd.concat([metadata, predict, predict_prob, feature_norm], axis=1)
    else:
        df = pd.concat([metadata, predict, feature_norm], axis=1)
    try:
        print("ATTEMPTING TO SAVE TO PATH: {}".format(out))
        output_path = os.path.join("/".join(out.split("/")[:-1]))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.to_csv(out, index=False)
    except (KeyError, FileNotFoundError, OSError):
        logging.error("Output folder creation failed.")
        exit(-1)


def impute_data(data):
    """Impute the given data.

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
    data_cols = data_np.columns.values
    data_np = data_np.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_np = imp_mean.fit_transform(data_np)
    # return to dataframe
    data = pd.DataFrame(data_np)  # removed a scaler fit-transform here
    data.columns = data_cols

    return data


def output_graph(data, predict, outfname):
    """Output a pairplot of the predicted values.

    Args:
        data (DataFrame): Data predicted on. Preferably normalized.
        predict (DataFrame): Class label predictions.
        out (str): Output file path/name

    """
    # change data in frame to be useable for graph
    df = pd.concat([data, predict], axis=1)
    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data.shape[1]],
        hue="predict",
        diag_kind='hist'
    )
    out.savefig("{}.png".format(outfname))


def save_clf_dialog(clf):
    """Present the user with the option to save the given classifier.

    Args:
        clf (Classifier): A trained classifier.

    """
    print("would you like to save the trained classifier? (y/n)")
    while 1:
        answer = input("> ")
        if answer in ["y", "yes"]:
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
            break
        elif answer in ["n", "no"]:
            break
        else:
            continue
