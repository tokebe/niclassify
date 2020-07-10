"""
A set of utilties for handling the creation and application of a classifier.

Strictly to be used in concert with the rest of the niclassify package.
"""
try:
    import csv
    import json
    import logging
    import os
    import re
    import subprocess
    import sys
    import xlrd
    import requests
    import tempfile

    import matplotlib as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from Bio.Align.Applications import MuscleCommandline
    from Bio.Phylo.Applications import RaxmlCommandline
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

    # using txt; must figure out delmiter
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

    # # replace unknown values with nan (should be redundant)
    # raw_data.replace(
    #     {
    #         "unknown": np.nan,
    #         "Unknown": np.nan,
    #         "0": np.nan  # replace string zero with nan (shouldn't exist)
    #     },
    #     inplace=True
    # )

    return raw_data  # return extracted data


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
    logging.info("extracting fully known data...")
    features = features.dropna()
    metadata = metadata.iloc[features.index]
    features.reset_index(drop=True, inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    # features and metadata now only contain rows with known class labels
    # as well as only fully known data (no NA feature values)
    return features, metadata


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
    data = pd.get_dummies(
        data,
        columns=category_cols,
        prefix=category_cols, drop_first=True)
    feature_cols = data.columns.values
    data = data.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp_mean.fit_transform(data)
    # return to dataframe
    data = pd.DataFrame(data)  # removed a scaler fit-transform here
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
    except (KeyError, FileNotFoundError, OSError, IOError):
        raise OSError("output folder creation failed.")

    # TODO keep trying to find ways to not use pd.concat


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
    data = pd.get_dummies(
        data,
        columns=category_cols,
        prefix=category_cols, drop_first=True)
    feature_cols = data.columns.values  # save column names
    data = data.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data))
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


def get_geo_taxon(filename, geo=None, taxon=None, api=None):
    """
    Save a request result from the api.

    Args:
        filename (str): Path to file to be created.
        geo (str): Geography descriptor
        taxon (str): Taxonomy descriptor
        api (str, optional): Base API URL. Defaults to None.

    Raises:
        OSError: If file creation fails.
        request.RequestException: If request otherwise fails.

    """
    print("making request...")
    if api is None:
        api = "http://www.boldsystems.org/index.php/API_Public/combined?"

    if not os.path.isabs(filename):
        filename = os.path.join(MAIN_PATH, "data/unprepared/" + filename)

    # create request from options
    request = []

    if taxon is not None:
        request.append("taxon={}".format(taxon))
    if geo is not None:
        request.append("geo={}".format(geo))
    request.append("format=tsv")

    request = api + "&".join(request)

    # try:
    with open(filename, "wb") as file, \
            requests.get(request, stream=True) as response:

        # error if response isn't success
        response.raise_for_status()

        # otherwise read to file
        for line in response.iter_lines():
            # print(line, end="\r")
            file.write(line)
            file.write(b"\n")

    # except (OSError, IOError, KeyError, TypeError, ValueError):
    #     raise OSError("File could not be created.")

    # except requests.RequestException:
    #     raise request.RequestException("Request Failed.")


def prep_sequence_data(data):
    """
    Prepare sequence data previously saved from API.

    Args:
        data (DataFrame): DataFrame of sequence data.

    Returns:
        DataFrame: The data after cleaning.

    Raises:
        pandas.errors.ParserError: If data could not be parsed. Likely caused
            by request returning extraneous error code.

    """
    if data.shape[0] == 0:
        raise ValueError("Datafile contains no observations.")

    # change to str in case it's not?
    data["nucleotides"] = data["nucleotides"].astype(str)

    # remove rows missing COI-5P in marker_codes
    data = data[data["marker_codes"].str.contains("COI-5P", na=False)]

    # remove rows with less than 350 base pairs
    data = data[
        data.apply(
            (lambda x: True
             if len([i for i in x["nucleotides"] if i.isalpha()]) >= 350
             else False),
            axis=1
        )
    ]

    return data


def write_fasta(data, filename):
    """
    Write a fasta file from a dataframe of sequence data.

    Args:
        data (DataFrame): sequence data, preferably filtered.
        filename (str): path to save fasta file to.
    """
    with open(filename, "w") as file:
        for index, row in data.iterrows():
            file.write(">{}\n".format(row["processid"]))
            file.write(row["nucleotides"])
            file.write("\n")


def align_fasta(infname, outfname, external=None):
    # TODO potentially support linux/mac
    # TODO or just change it to stdout in a tkinter for easier support
    """
    Generate an alignment for the given fasta file.

    Args:
        infname (str): Path to fasta to be aligned.
        outfname (str): Path to output fasta to be
    """
    alignment_call = MuscleCommandline(
        os.path.realpath(
            os.path.join(MAIN_PATH, "bin/muscle3.8.31_i86win32.exe")
        ),
        input=os.path.realpath(infname),
        out=os.path.realpath(outfname)
    )

    print(alignment_call.__str__())

    if external is not None:
        subprocess.run(
            alignment_call.__str__(),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.Popen(alignment_call.__str__())


def delimit_species_GMYC(infname, outfname, external=None):
    r_script_exe = os.path.realpath(
        os.path.join(
            MAIN_PATH, "bin/R/R-Portable/App/R-Portable/bin/Rscript.exe")
    )
    r_script = os.path.realpath(
        os.path.join(
            MAIN_PATH, "niclassify/core/delim_spec.r")
    )

    if external is not None:
        subprocess.run(
            '"{}" "{}" "{}" "{}"'.format(
                r_script_exe,
                r_script,
                infname,
                outfname
            ),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.run(
            '"{}" "{}" "{}" "{}"'.format(
                r_script_exe,
                r_script,
                infname,
                outfname
            )
        )


def delimit_species_PTP(infname, outfname, external=None):

    # STEPS
    # create tree from raxml (save in a tempdir)
    #   you can then access the file with name RAxML_bestTree.<name>
    # feed into ptp (save the resulting delimitation)

    # make a tempdir for the RAxML output
    raxfolder = tempfile.TemporaryDirectory()

    phylo_call = \
        "bin/raxmlHPC.exe -m GTRGAMMA -p 12345 -s {} -n rax -w {}".format(
            infname, raxfolder.name)

    tree = os.path.join(raxfolder.name, "RAxML_bestTree.rax")

    # TODO looks like you have to package python2 to make this work
    # we can get back to this later
