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

    from sklearn import preprocessing
    from sklearn.cluster import KMeans
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn import metrics
    # from sklearn.model_selection import KFold

    from itertools import chain, combinations
except ModuleNotFoundError:
    print("Missing required modules. Install requirements by running")
    print("'python -m pip install -r requirements.txt'")


sns.set()

# ----- Extra Configuration ----

N_TRIALS = 10  # number of folds for KFold x-val

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


def create_kmeans(x, k, weights=None):
    """Create list of cluster ID's using k-means clustering.

    Args:
        x (ndarray): Data to cluster.
        k (int): number of clusters to make

    Returns:
        array: cluster ID's

    """
    # kmeans = make_pipeline(
    #     StandardScaler().fit_transform(),
    #     KMeans(n_clusters=k)
    # )
    kmeans = KMeans(n_clusters=k)

    return kmeans.fit(x, sample_weight=weights)


def get_predict(cluster, metadata_known, c_label="Status"):
    """Get the predicted labels from cluster assignments and known labels.

    Args:
        cluster (Series): A series of cluster assignments.
        metadata_known (DataFrame): Metadata (including class labels) of known
            data.
        c_label (str, optional): Name of column containing class labels.
            Defaults to "Status".

    Returns:
        [type]: [description]

    """
    matches = {}
    # loop over metadata rows
    # add number of matches to either cluster 0 or 1 per status label
    for index, row in metadata_known.iterrows():

        # avoid registering nan as a match label
        if ((type(row[c_label]) == str)
                and (row[c_label] not in matches.keys())):
            matches[row[c_label]] = [0, 0]
        try:  # increment number of matches to cluster label
            matches[row[c_label]][cluster[index]] += 1
        except KeyError:  # catches nan values
            continue

    # print(matches)

    # convert frequencies to proportions
    for key, val in matches.copy().items():
        matches[key] = [x / sum(val) for x in val]
    # print(matches)
    m = 0  # the highest proportion of selected values in either cluster
    lab = ""  # label with the highest prop. of selected vals in either cluster
    # change m to cluster number with highest prop. of a key.
    for key, val in matches.items():
        m_v = max(val)
        if m_v > m:
            m = m_v
            lab = key
    n = [x for x in matches.keys() if x != lab][0]  # other label
    l_selector = 0 if matches[lab][0] > matches[lab][1] else 1
    # generate predicted label map
    # predict = pd.Series(
    #     [lab if x == l_selector else n for x in cluster],
    #     name="predict", dtype="object")
    predict = {l_selector: lab, 0 if l_selector == 1 else 1: n}

    return predict


def powerset(iterable):
    """Return the powerset of an iterable, without empty set.

    Args:
        iterable (iterable): A list or other iterable.

    Returns:
        set: a set of sets, the powerset of given iterable.

    """
    # courtesy of itertools documentation
    # modified not to return empty set
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def get_misclass_rates(metadata_known, c_label, predict):
    nerr = metadata_known[
        (metadata_known[c_label] == "native") & (
            predict == "introduced")
    ].shape[0] / metadata_known[
        metadata_known[c_label] == "native"].shape[0] * 100
    ierr = metadata_known[
        (metadata_known[c_label] == "introduced") & (
            predict == "native")
    ].shape[0] / metadata_known[
        metadata_known[c_label] == "introduced"].shape[0] * 100

    return nerr, ierr


def test_features(data_known, metadata_known, c_label="Status", w_label=None):

    # run each set and get the mean ba for 10 runs
    ba = []
    nerr = []
    ierr = []
    # TODO turn this into its own function to re-use in test
    for i in range(N_TRIALS):
        # make cluster
        km = create_kmeans(
            data_known,
            k=2,
            weights=None if w_label is None else 1 / metadata_known[w_label])
        cluster = km.predict(
            data_known,
            sample_weight=(None
                           if w_label is None
                           else 1 / metadata_known[w_label]))
        # get cluster assignments
        cluster_assign = get_predict(cluster, metadata_known, c_label)
        # get predictions
        predict = pd.Series(cluster).replace(cluster_assign)
        # add balanced accuracy to list
        ba.append(metrics.balanced_accuracy_score(
            metadata_known[c_label][metadata_known[c_label].notnull()],
            predict[metadata_known[c_label].notnull()]))
        nerr1, ierr1 = get_misclass_rates(metadata_known, c_label, predict)
        nerr.append(nerr1)
        ierr.append(ierr1)

    return np.mean(ba), np.mean(nerr), np.mean(ierr)


def get_best_features_ba(data_known, metadata_known, c_label="Status",
                         w_label=None):
    """Get features which give best balanced accuracy predicting known values.

    Args:
        data_known (DataFrame): data for known class labels used for
            classification
        metadata_known (DataFrame): metadata with known class labels
        c_label (str, optional): column name for class labels in metadata.
            Defaults to "Status".
        w_label (str, optional): column name for weight values.
            Defaults to None.

    Returns:
        list: a list of optimal feature column names

    """
    best_ba = []
    bestba = 0
    nerr = 0
    ierr = 0
    pset = powerset(data_known.columns)  # all possible feature combos
    fset_len = len(set(pset))  # len of power set; n combos
    fset = 1  # counter for output
    # look at every possible combination
    for s in powerset(data_known.columns):
        if fset < fset_len:
            print("  test set {} / {}".format(fset, fset_len), end='\r')
        else:
            print("  test set {} / {}".format(fset, fset_len))
        fset += 1

        mean_ba, mean_nerr, mean_ierr = test_features(
            data_known[list(s)],
            metadata_known,
            c_label,
            w_label)
        # update best features if new mean ba is better
        if mean_ba > bestba:
            bestba = mean_ba
            best_ba = list(s)
            nerr = mean_nerr
            ierr = mean_ierr
    # print(nerr)
    # print out results
    print("Found best balanced accuracy of {}".format(bestba))
    print("Percent of Native mislabled to Introduced: {:.2f}%".format(
        nerr))
    print("Percent of Introduced mislabled to Native: {:.2f}%".format(
        ierr))
    print("Using features:")
    for f in best_ba:
        print("  {}".format(f))

    return best_ba  # return list of best features


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
        "--weight",
        nargs='?',
        default=None,
        help="column name to be used for weighted clustering, defaults to N")
    parser.add_argument(
        "-n",
        "--nanval",
        nargs='*',
        help="additional values to be counted as NA/NaN")
    parser.add_argument(
        "-p",
        "--predicton",
        nargs=2,
        default=None,
        help="Use a previously saved random forest to predict on data, \
            specified by given path/filename for classifier AND features json")
    parser.add_argument(
        "-k",
        "--knownset",
        action="store_true",
        help="use only known cases for training"
    )

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

    # print(raw_data)

    # get feature data column range
    if not re.match("[0-9]+:[0-9]+", args.dcol):
        parser.error("data column selection range format invalid (see -h).")
    else:
        dcol = args.dcol.split(":")
        dcol = [int(x) for x in dcol]
        dcol[0] -= 1
        dcol[1] += 1

    # drop rows with na count. migh need to change for more general program.
    if args.weight is not None:
        raw_data = raw_data[raw_data[args.weight].notnull()]
        raw_data.reset_index(drop=True, inplace=True)

    # replace unknown values with nan
    raw_data.replace({
        "unknown": np.nan,
        "Unknown": np.nan,
        "0": np.nan})

    # split data into feature data and metadata
    data = raw_data.iloc[:, dcol[0]:dcol[1] + 1]
    metadata = raw_data.drop(raw_data.columns[dcol[0]:dcol[1]], axis=1)

    # convert class labels to lower
    metadata.loc[:, args.clabel] = metadata[args.clabel].str.lower()

    # debug
    # print(metadata["Final Status"].unique())
    # print(data.head())
    # print(metadata.head())

    # impute data
    print("imputing data...")
    data_np = data.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_np = imp_mean.fit_transform(data_np)

    # scale data
    print("scaling data...")
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = data.columns

    # split into known and unknown
    print("splitting data...")
    data_known = data_norm[metadata[args.clabel].notnull()]
    metadata_known = metadata[metadata[args.clabel].notnull()]
    # reset indices
    data_known.reset_index(drop=True, inplace=True)
    metadata_known.reset_index(drop=True, inplace=True)

    # print(data_known)

    # debug
    # print(data_known)
    # print(metadata_known)

    # obtain best features if preset is not provided (and chosen)
    if args.predicton is None:

        # get best features for known data. Potentially subject to overfitting.
        print("obtaining best features...")
        if args.knownset:
            best_features = get_best_features_ba(
                data_known, metadata_known, args.clabel, args.weight)
        else:
            best_features = get_best_features_ba(
                data_norm, metadata, args.clabel, args.weight)

        print("classifying all data...")
        km = create_kmeans(
            data_known,
            k=2,
            weights=(None
                     if args.weight is None
                     else 1 / metadata_known[args.weight]))
        cluster = km.predict(
            data_known,
            sample_weight=(None
                           if args.weight is None
                           else 1 / metadata_known[args.weight]))
        # get cluster assignments
        cluster_assign = get_predict(cluster, metadata_known, args.clabel)
        # get predictions
        predict = pd.Series(cluster).replace(cluster_assign)

    # otherwise use saved features list
    else:
        with open(args.predicton[1], "r") as preset:
            bf_json = json.load(preset)
            best_features = bf_json.get("best features")
            cluster_assign = bf_json.get("cluster assign")
            from joblib import load
            km = load(args.predicton)
            cluster = km.predict(
                data_known,
                sample_weight=(None
                               if args.weight is None
                               else 1 / metadata_known[args.weight]))
            predict = pd.Series(cluster).replace(cluster_assign)
        exit()

    # save output
    print("saving new output...")
    df = pd.concat([metadata, predict, data], axis=1)
    try:
        df.to_csv(args.out, index=False)
    except (KeyError, FileNotFoundError):
        parser.error("intended output folder does not exist!")

    # change data in frame to be useable for graph
    del df
    df = pd.concat([data_norm[best_features], predict], axis=1)

    # generate and output graph
    print("generating graphs...")
    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data_norm[best_features].shape[1]],
        hue="predict"
    )
    out.savefig("{}.png".format(args.out))

    fig, ax = plt.pyplot.subplots(nrows=1, ncols=1)
    metrics.plot_confusion_matrix(
        km,
        data_known,
        metadata_known[args.clabel],
        ax=ax,
        normalize="true")
    ax.grid(False)
    fig.savefig("{}.cm.png".format(args.out))

    print("...done!")

    if args.predicton is None:

        print("would you like to save the trained classifier? (y/n)")
        while 1:
            answer = input("> ")
            if answer in ["y", "yes"]:
                if not os.path.exists("output/forests"):
                    os.makedirs("output/forests")
                from joblib import dump
                i = 0
                while os.path.exists(
                        "output/classifiers/classifier{}.joblib".format(i)):
                    i += 1
                dump(km, "output/classifiers/kmeans{}.joblib".format(i))
                # save best features to json file
                bf_json = json.dumps(
                    {"best features": best_features,
                     "cluster assign": cluster_assign},
                    indent=4)
                i = 0
                while os.path.exists(
                        "output/classifiers/km_feat{}.joblib".format(i)):
                    i += 1
                with open(
                    "output/classifiers/km_feat{}.joblib".format(i), "w") \
                        as preset:
                    preset.write(bf_json)
                break
            elif answer in ["n", "no"]:
                break
            else:
                continue


if __name__ == "__main__":
    main()
    # ----- Things I would like to add for completeness: -----
    # TODO save the classifier (dump) / read in saved classifier
    # TODO
