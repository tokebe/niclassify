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


sns.set()

NANS = [
    "nan",
    "NA",
    '',
    ' # N/A',
    '#N/A N/A',
    '#NA',
    '-1.#IND',
    '-1.#QNAN',
    '-NaN',
    '-nan',
    '1.#IND',
    '1.#QNAN',
    '<NA>',
    'N/A',
    'NA',
    'NULL',
    'NaN',
    'n/a',
    'nan',
    'null',
    '#DIV/0!',
    "unknown",
    "Unknown"
]


def train_forest(
        data_known,
        metadata_known,
        c_label="Status",
        multirun=1):
    print("  obtaining best hyperparameters...")

    x_train, x_test, y_train, y_test = train_test_split(
        data_known, metadata_known[c_label],
        stratify=metadata_known[c_label],
        test_size=0.2)  # previously .15

    best_train_predict = None
    best_test_predict = None
    best_train_ba = 0
    best_test_ba = 0
    best_model = None

    for i in range(multirun):

        print(
            "    testing forest {} of {}...".format(i + 1, multirun), end="\r")

        # hyperparameters to optimize
        parameters = {
            'n_estimators': np.linspace(500, 1000).astype(int),
            'max_depth': list(np.linspace(10, 20).astype(int)),

            'min_samples_split': list(np.arange(0.05, 0.5, 0.05))}

        rf = RandomForestClassifier(class_weight="balanced", oob_score=True)

        # Create the random search
        rs = RandomizedSearchCV(
            rf,
            parameters,
            n_jobs=-1,
            scoring='balanced_accuracy',
            cv=StratifiedKFold(n_splits=10))

        rs.fit(x_train, y_train)

        train_predict = rs.best_estimator_.predict(x_train)
        test_predict = rs.best_estimator_.predict(x_test)

        train_ba = metrics.balanced_accuracy_score(y_train, train_predict)
        test_ba = metrics.balanced_accuracy_score(y_test, test_predict)

        if test_ba > best_test_ba:
            best_train_predict = train_predict
            best_test_predict = test_predict
            best_train_ba = train_ba
            best_test_ba = test_ba
            best_model = rs.best_estimator_

    print("\n")

    print("  found best hyperparameters (as follows):")
    for key, val in rs.best_params_.items():
        print("    {}: {}".format(key, val))

    print("out-of-bag score: {}".format(best_model.oob_score_))
    print("---")

    train_nerr = (y_train[
        (y_train == "native")
        & (best_train_predict == "introduced")].shape[0]
        / y_train[
            y_train == "native"].shape[0]) * 100
    train_ierr = (y_train[
        (y_train == "introduced")
        & (best_train_predict == "native")].shape[0]
        / y_train[
            y_train == "introduced"].shape[0]) * 100

    test_nerr = (y_test[
        (y_test == "native")
        & (best_test_predict == "introduced")].shape[0]
        / y_test[
            y_test == "native"].shape[0]) * 100
    test_ierr = (y_test[
        (y_test == "introduced")
        & (best_test_predict == "native")].shape[0]
        / y_test[
            y_test == "introduced"].shape[0]) * 100

    print("train set BA: {}".format(best_train_ba))
    print(
        "train set: Percent of Native mislabled to Introduced: {:.2f}%".format(
            train_nerr))
    print(
        "train set: Percent of Introduced mislabled to Native: {:.2f}%".format(
            train_ierr))
    print("---")
    print("test set BA : {}".format(best_test_ba))
    print(
        "test set: Percent of Native mislabled to Introduced: {:.2f}%".format(
            test_nerr))
    print(
        "test set: Percent of Introduced mislabled to Native: {:.2f}%".format(
            test_ierr))

    return best_model


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


def main():
    """Run the program.

    Takes in user arguments to select data to run on, and outputs a new csv
    with predictions.
    """
    # get args
    parser = argparse.ArgumentParser()
    args = getargs(parser)

    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/tree_images"):
        os.makedirs("output/tree_images")

    # get raw data
    if ".xlsx" in args.data[-5:]:
        if args.excel is not None:
            if args.excel.isdigit():
                raw_data = pd.read_excel(
                    args.data,
                    sheet_name=int(args.excel) - 1,
                    na_values=NANS,
                    keep_default_na=True)
            else:
                raw_data = pd.read_excel(
                    args.data,
                    sheet_name=args.excel,
                    na_values=NANS,
                    keep_default_na=True)
    elif ".csv" in args.data[-4:]:
        raw_data = pd.read_csv(args.data, na_values=NANS, keep_default_na=True)
    elif ".tsv" in args.data[-4:]:
        raw_data = pd.read_csv(
            args.data,
            na_values=NANS,
            keep_default_na=True,
            sep="\t")
    elif ".txt" in args.data[-4:]:
        raw_data = pd.read_csv(
            args.data,
            na_values=NANS,
            keep_default_na=True,
            sep=None)
    else:
        parser.error(
            "data file type is unsupported, or file extension not included")

    # get feature data column range
    if not re.match("[0-9]+:[0-9]+", args.dcol):
        parser.error("data column selection range format invalid (see -h).")
    else:
        dcol = args.dcol.split(":")
        dcol = [int(x) for x in dcol]
        dcol[0] -= 1
        dcol[1] += 1

    # replace unknown values with nan
    raw_data.replace({
        "unknown": np.nan,
        "Unknown": np.nan,
        "0": np.nan})

    # remove instances of >1 group count
    raw_data = raw_data.loc[raw_data["Group_Count"] <= 1]
    raw_data.reset_index(drop=True, inplace=True)

    # split data into feature data and metadata
    data = raw_data.iloc[:, dcol[0]:dcol[1]]
    metadata = raw_data.drop(raw_data.columns[dcol[0]:dcol[1]], axis=1)

    # print(metadata.columns)

    # convert class labels to lower
    metadata.loc[:, args.clabel] = metadata[args.clabel].str.lower()

    # scale data
    print("scaling data...")
    # get categorical columns for dummy variable encoding
    cols_cat = list(data.select_dtypes(exclude=[np.number]).columns.values)
    data_np = pd.get_dummies(
        data,
        columns=cols_cat,
        prefix=cols_cat, drop_first=True)
    data_cols = data_np.columns.values
    data_np = data_np.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = data_cols

    # stuff for training classifier if none is provided
    if args.predicton is None:
        # split into known and unknown
        print("splitting data...")
        data_known = data_norm[metadata[args.clabel].notnull()]
        metadata_known = metadata[metadata[args.clabel].notnull()]
        # reset indices
        data_known.reset_index(drop=True, inplace=True)
        metadata_known.reset_index(drop=True, inplace=True)

        # print(data_norm)

        # remove null rows from known data
        print("extracting fully known data...")
        data_known = data_known.dropna()
        metadata_known = metadata_known.iloc[data_known.index]
        data_known.reset_index(drop=True, inplace=True)
        metadata_known.reset_index(drop=True, inplace=True)

        v = metadata_known[args.clabel].value_counts()
        print("  {} {}\n  {} {}".format(
            v[0],
            v.index[0],
            v[1],
            v.index[1]))

        # get classifier (train it)
        print("training random forest...")
        forest = train_forest(data_known, metadata_known,
                              args.clabel, args.multirun)

        print("saving confusion matrix...")
        fig, ax = plt.pyplot.subplots(nrows=1, ncols=1)
        metrics.plot_confusion_matrix(
            forest,
            data_known,
            metadata_known[args.clabel],
            ax=ax,
            normalize="true")
        ax.grid(False)
        fig.savefig("{}.cm.png".format(args.out))
    # stuff for opening provided classifier
    else:
        from joblib import load
        forest = load(args.predicton)

    # impute data
    print("imputing data...")
    # get categorical columns for dummy variable encoding
    cols_cat = list(data_norm.select_dtypes(
        exclude=[np.number]).columns.values)
    data_np = pd.get_dummies(
        data_norm,
        columns=cols_cat,
        prefix=cols_cat, drop_first=True)
    data_cols = data_np.columns.values
    data_np = data_np.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_np = imp_mean.fit_transform(data_np)
    # return to dataframe
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = data_cols

    # make predictons
    predict = pd.DataFrame(forest.predict(data_norm))
    predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)
    predict_prob = pd.DataFrame(forest.predict_proba(data_norm))
    predict_prob.rename(
        columns={predict_prob.columns[0]: "predict prob"}, inplace=True)


    # save output
    print("saving new output...")
    df = pd.concat([metadata, predict, predict_prob, data_norm], axis=1)
    try:
        df.to_csv(args.out, index=False)
    except (KeyError, FileNotFoundError):
        parser.error("intended output folder does not exist!")

    # change data in frame to be useable for graph
    del df
    df = pd.concat([data_norm, predict], axis=1)

    # generate and output graph
    print("generating final graphs...")
    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data_norm.shape[1]],
        hue="predict"
    )
    out.savefig("{}.png".format(args.out))

    print("...done!")

    if args.predicton is None:

        print("would you like to save the trained classifier? (y/n)")
        while 1:
            answer = input("> ")
            if answer in ["y", "yes"]:
                if not os.path.exists("output/classifiers"):
                    os.makedirs("output/classifiers")
                if not os.path.exists("output/classifiers/forests"):
                    os.makedirs("output/classifiers/forests")
                from joblib import dump
                i = 0
                while os.path.exists("output/classifiers/forests/forest{}.joblib".format(i)):
                    i += 1
                dump(forest, "output/classifiers/forests/forest{}.joblib".format(i))
                break
            elif answer in ["n", "no"]:
                break
            else:
                continue


if __name__ == "__main__":
    main()
