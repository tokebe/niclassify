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

    return kmeans.fit_predict(x, sample_weight=weights)


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
    # generate predicted labels
    predict = pd.Series(
        [lab if x == l_selector else n for x in cluster],
        name="predict", dtype="object")

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


def kmeans_graph(data, metadata, filename, c_label="Status", weights=None):
    """Plot the results of a kmeans classification and save it.

    Args:
        data (DataFrame): Data to be used for classification
        metadata (DataFrame): Metadata including class labels
        filename (str): name of image to save
        c_label (str, optional): column name for class labels in metadata.
            Defaults to "Status".
        weights (Series, optional): Weights to use in clustering.
             Defaults to None.
    """
    cluster = create_kmeans(data, 2, weights=weights)  # make cluster
    predict = get_predict(cluster, metadata, c_label)  # get predictions

    df = pd.concat([data, predict], axis=1)

    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data.shape[1]],
        hue="predict"
    )
    out.savefig("output/{}.png".format(filename))


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
        cluster = create_kmeans(
            data_known,
            k=metadata_known[c_label].nunique(),
            weights=None if w_label is None else 1 / metadata_known[w_label])
        # get predictions from cluster assignments
        predict = get_predict(cluster, metadata_known, c_label)
        # add balanced accuracy to list
        ba.append(metrics.balanced_accuracy_score(
            metadata_known[c_label], predict))
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
            bestba = np.mean(mean_ba)
            best_ba = list(s)
            nerr = mean_nerr,
            ierr = mean_ierr

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
        "--preset",
        action="store_true",
        help="use saved json file with preset features instead of obtaining \
            best features."
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="test accuracy and misclassification error \
        of currently saved preset features"
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
    else:
        parser.error(
            "data file type is unsupported, or file extension not included")

    # print(raw_data)

    # get feature data column range
    if not re.match("[0-9]:[0-9]", args.dcol):
        parser.error("data column selection range format invalid (see -h).")
    else:
        dcol = args.dcol.split(":")
        dcol = [int(x) - 1 for x in dcol]

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
    data = raw_data.iloc[:, dcol[0]:dcol[1]]
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

    # just test saved features (if argument to do so given)
    if args.test:
        print("testing saved features...")
        with open("output/best_features.json", "r") as preset:
            bf_json = json.load(preset)
            best_features = bf_json.get("best features")

        mean_ba, mean_nerr, mean_ierr = test_features(
            data_known[best_features],
            metadata_known,
            args.clabel,
            args.weight)

        print("Balanced accuracy of {}".format(mean_ba))
        print("Percent of Native mislabled to Introduced: {:.2f}%".format(
            np.mean(mean_nerr)))
        print("Percent of Introduced mislabled to Native: {:.2f}%".format(
            np.mean(mean_ierr)))
        print("Using features:")
        for f in best_features:
            print("  {}".format(f))

        exit()

    # obtain best features if preset is not provided (and chosen)
    if not args.preset:

        # get best features for known data. Potentially subject to overfitting.
        print("obtaining best features...")
        best_features = get_best_features_ba(
            data_known, metadata_known, args.clabel, args.weight)

        # save best features to json file
        bf_json = json.dumps({"best features": best_features}, indent=4)
        with open("output/best_features.json", "w") as preset:
            preset.write(bf_json)

        # create graph to show training clusters
        print("generating training graph...")
        kmeans_graph(
            data_known[best_features],
            metadata_known,
            "md_cluster_training",
            args.clabel,
            weights=1/metadata_known[args.weight])
    # otherwise use saved features list
    else:
        with open("output/best_features.json", "r") as preset:
            bf_json = json.load(preset)
            best_features = bf_json.get("best features")

    # debug
    # best_features = ["dN_Distance",
    #                  "dS_Distance",
    #                  "AA_Distance",
    #                  "dN/dS_Distance"]

    # create kmeans on all data using selected features
    print("classifying all data...")
    cluster = create_kmeans(
        data_norm[best_features],
        k=2,
        weights=1 / metadata[args.weight])
    predict = get_predict(cluster, metadata, args.clabel)

    # save output
    print("saving new output...")
    df = pd.concat([metadata, predict, data], axis=1)
    try:
        df.to_csv(args.out)
    except (KeyError, FileNotFoundError):
        parser.error("intended output folder does not exist!")

    # change data in frame to be useable for graph
    del df
    df = pd.concat([data_norm[best_features], predict], axis=1)

    # generate and output graph
    print("generating final graph...")
    out = sns.pairplot(
        data=df,
        vars=df.columns[0:data_norm[best_features].shape[1]],
        hue="predict"
    )
    out.savefig("output/{}.png".format("md_cluster_inc_unknown"))

    print("...done!")


if __name__ == "__main__":
    main()
    # ----- Things I would like to add for completeness: -----
    # TODO better error checking (are colname arguments valid? etc)
    # TODO increased and clear comments and documentation
    # TODO add argument for label to minimize error for, instead of max'ing BA
    # TODO add logging of printed output
    # TODO add optional debug log which only writes to file
    # TODO add optional argument to not impute data
    # TODO add option to test ba on full set, and handling, default to full set
