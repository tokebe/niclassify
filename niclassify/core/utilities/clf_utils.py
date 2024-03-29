"""
Utilities for data, file, and program interactions relating to training and
using an AutoClassifier.

Generally you want to import by importing the directory, utilities, and
accessing by utilities.function (instead of utilities.general_utils.function).
"""

import logging
import os
import xlrd

import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from ..classifiers import AutoClassifier
from .general_utils import USER_PATH

sns.set()


def get_known(features, metadata, class_column):
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
    features = features[metadata[class_column].notnull()]
    metadata = metadata[metadata[class_column].notnull()]
    # reset indices
    features.reset_index(drop=True, inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    # remove null rows from known data
    logging.info("extracting known data...")
    NA_rows = features.index[features.isnull().all(1)]  # rows w/ all NA
    NA_cols = features.columns[features.isnull().all()]  # cols w/ all NA

    if len(NA_cols) > 0:
        logging.warning(
            f"removing column{'s' if len(NA_cols) > 1 else ''} due to all NA values: {' '.join(NA_cols.to_list())}"
        )
        features = features.drop(NA_cols, axis=1)
    if len(NA_rows) > 0:
        if 'UPID' in metadata.columns:
            logging.warning(
                f"removing row{'s' if len(NA_rows) > 1 else ''} due to all NA values {' '.join(metadata.iloc[NA_rows]['UPID'].to_list())} "
            )
        else:
            logging.warning(
                f"removing {len(NA_rows)} row{'s' if len(NA_rows) > 1 else ''} due to all NA values"
            )
        features = features.drop(NA_rows, axis=0)
    metadata = metadata.iloc[features.index]
    features.reset_index(drop=True, inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    # features and metadata now only contain rows with known class labels
    # as well as only rows/columns with at least partial data
    return features, metadata


def impute_data(data):
    """
    Impute the given data.

    Args:
        data (DataFrame): Data. Preferably normalized.

    Returns:
        DataFrame: The imputed data.

    """
    # drop completely empty columns
    data = data.dropna(how="all", axis=1)

    # get column order for order preservation
    col_order = data.columns.values

    # get categorical columns
    category_cols = list(data.select_dtypes(
        exclude=[np.number]).columns.values)

    # split data into categorical and numerical
    categorical = data[category_cols]
    data = data.drop(columns=category_cols)

    # impute categorical by frequency
    categorical = categorical.apply(
        lambda col: col.fillna(col.value_counts().index[0])
    )

    feature_cols = data.columns.values
    data = data.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp_mean.fit_transform(data)
    # return to dataframe
    data = pd.DataFrame(data)  # removed a scaler fit-transform here
    data.columns = feature_cols

    # combine imputed categorical with numerical
    if categorical.shape[1] > 0:
        data = pd.concat([data, categorical], axis='columns')

    # reorder to match original
    data = data[col_order]

    return data


def load_classifier(filename):
    """
    Load a saved classifier.

    Args:
        filename (str): Path/name of classifier.

    Returns:
        Classifier: The loaded classifier.

    """
    if not os.path.isabs(filename):
        filename = os.path.join(USER_PATH, "output/classifiers/", filename)

    # check if filename exists
    if not os.path.exists(filename):
        raise ValueError("file {} does not exist.".format(filename))

    from joblib import load
    classifier = load(filename)

    if not isinstance(classifier, AutoClassifier):
        raise TypeError("loaded file is not compatible (not AutoClassifier)")

    return classifier


def make_confm(clf, features_known, class_labels):
    """
    Make a confusion matrix plot of a trained classifier.

    Args:
        clf (AutoClassifier): An AutoClassifier.
        data_known (DataFrame): Data for which class labels are known.
        class_labels (Series): Known class labels.

    Returns:
        figure: A matplotlib figure containing the graph.

    """
    # type error checking
    if type(features_known) is not pd.DataFrame:
        raise TypeError("Cannot save: features_known is not DataFrame.")
    if (type(class_labels) is not pd.DataFrame
            and type(class_labels) is not pd.Series):
        raise TypeError("Cannot save: metadata_known is not DataFrame.")

    features_known = impute_data(features_known)

    fig, ax = plt.pyplot.subplots(nrows=1, ncols=1)
    metrics.plot_confusion_matrix(
        clf,
        features_known,
        class_labels,
        ax=ax,
        normalize="true")
    ax.grid(False)

    return fig


def make_pairplot(data, predict):
    """
    Make a pairplot figure.

    Args:
        data (DataFrame): Data predicted on. Preferably normalized.
        predict (DataFrame): Class label predictions.

    Returns:
        figure: The resulting pairplot.

    """
    # type error checking
    if type(predict) is not pd.DataFrame and type(predict) is not pd.Series:
        raise TypeError("Cannot save: predict is not DataFrame or Series.")

    # change data in frame to be useable for graph
    df = pd.concat([data, predict], axis=1)

    print("attempting to generate plot")
    # make the pairplot using seaborn
    pairplot = sns.pairplot(
        data=df,
        vars=df.columns[0:data.shape[1]],
        hue="predict",
        diag_kind='hist'
    )

    return pairplot


def save_clf(clf):
    """Save a given AutoClassifier.

    Args:
        clf (AutoClassifier): A trained classifier.
    """
    from joblib import dump
    i = 0
    while os.path.exists(
        os.path.join(
            USER_PATH,
            "output/classifiers/{}{}.gz".format(
                clf.__class__.__name__, i)
        )
    ):
        i += 1
    dump(
        clf,
        os.path.join(
            USER_PATH,
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


def save_confm(clf, features_known, class_labels, out):
    """
    Save a confusion matrix plot of a trained classifier.

    Args:
        clf (AutoClassifier): An AutoClassifier.
        data_known (DataFrame): Data for which class labels are known.
        class_labels (Series): Known class labels.
        out (str): file path/name for output image.
    """
    # type error checking
    if type(features_known) is not pd.DataFrame:
        raise TypeError("Cannot save: features_known is not DataFrame.")
    if (type(class_labels) is not pd.DataFrame
            and type(class_labels) is not pd.Series):
        raise TypeError("Cannot save: metadata_known is not DataFrame.")

    fig = make_confm(clf.clf, features_known, class_labels)
    if not os.path.isabs(out):
        out = os.path.join(USER_PATH, "output/" + out)
    fig.savefig("{}.cm.png".format(out))


def save_pairplot(data, predict, outfname):
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

    # make pairplot
    out = make_pairplot(data, predict)
    # save pairplot
    if not os.path.isabs(outfname):
        outfname = os.path.join(USER_PATH, "output/" + outfname)
    out.savefig("{}.png".format(outfname))


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
            out = os.path.join(USER_PATH, "output/" + out)

        output_path = os.path.join(
            "/".join(out.replace("\\", "/").split("/")[:-1]))

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("  saving files to path: {}".format(
            os.path.realpath(output_path)))
        df.to_csv(out, index=False)
    except (KeyError, FileNotFoundError, OSError, IOError):
        raise OSError("output folder creation failed.")


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
   # drop completely empty columns
    data.dropna(how="all", axis=1, inplace=True)

    # get column order for order preservation
    col_order = data.columns.values

    # get categorical columns
    category_cols = list(data.select_dtypes(
        exclude=[np.number]).columns.values)

    # split data into categorical and numerical
    categorical = data[category_cols]
    data = data.drop(columns=category_cols)

    feature_cols = data.columns.values  # save column names
    data = data.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    data.columns = feature_cols  # add column names back

    # combine imputed categorical with numerical
    if categorical.shape[1] > 0:
        data = pd.concat([data, categorical], axis='columns')

    # reorder to match original
    data = data[col_order]

    return data
