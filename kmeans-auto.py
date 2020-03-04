import sys

import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics

from itertools import chain, combinations

sns.set()


def get_data(filename):
    return pd.read_csv(filename)


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


def powerset(iterable):
    # courtesy of itertools documentation
    # modified not to return empty set
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def kmeans_graph(df, name, weights=None):
    test = df
    # print(test.iloc[:, 4:].head())
    test.loc[:, "Cluster"] = create_kmeans(test.iloc[:, 4:], 2, weights)

    nmatch1 = test[(test["Status"] == "native") &
                   (test["Cluster"] == 1)].shape[0]
    nmatch0 = test[(test["Status"] == "native") &
                   (test["Cluster"] == 0)].shape[0]
    imatch1 = test[(test["Status"] == "introduced") &
                   (test["Cluster"] == 1)].shape[0]
    imatch0 = test[(test["Status"] == "introduced") &
                   (test["Cluster"] == 0)].shape[0]

    nmatch_choice = 1 if nmatch1 > nmatch0 else 0
    test.loc[:, "Predict"] = ["native" if x ==
                              nmatch_choice else "introduced" for x in test["Cluster"]]
    test.loc[:, "Correct"] = test.apply(
        lambda row: 1 if row["Status"] == row["Predict"] else 0, axis=1)

    acc = 100 * test["Correct"].sum() / test["Correct"].shape[0]
    nerr = test[(test["Status"] == "native") & (test["Correct"] == 0)
                ].shape[0] / test[test["Status"] == "native"].shape[0] * 100
    ierr = test[(test["Status"] == "introduced") & (test["Correct"] == 0)
                ].shape[0] / test[test["Status"] == "introduced"].shape[0] * 100

    print("Balanced accuracy score:                   {:.2f}".format(
        metrics.balanced_accuracy_score(test["Status"], test["Predict"])))
    print("Percent of Predictions correct:            {:.2f}%".format(acc))
    print("Percent of Native mislabled to Introduced: {:.2f}%".format(nerr))
    print("Percent of Introduced mislabled to Native: {:.2f}%".format(ierr))

    out = sns.pairplot(
        data=df,
        vars=df.columns[4:-3],
        hue="Predict"
    )
    out.savefig("plots/{}.png".format(name))


# TODO complete basic feature selection
# TODO include cross-validation mean ba
# TODO generalize to work with any number of status levels
def get_best_features_ba(data_known, metadata_known):
    best_ba = []
    bestba = 0
    for s in powerset(data_known.columns):
        cluster = create_kmeans(
            data_known[list(s)],
            k=metadata_known["Status"].nunique(),
            weights=1 / metadata_known["N"])

        matches = {}


# TODO redo argument system to allow designation of column for class prediciton
# TODO redo argument system to fall in line with most runtime-argument design
def main(args):
    """Generate a new datasheet with predictions.

    runtime args:
        python kmeans-auto.py a b c
        a: data filename
        b: first column number of features (1-indexed)
        c: output filename (optional)

    Args:
        args (str): list of runtime arguments

    """
    # get data
    raw_data = get_data(args[0])
    col = args[1] - 1
    metadata = raw_data.iloc[:, :col]
    data = raw_data.iloc[:, col:]
    # impute data
    data_np = data.to_numpy()
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_np = imp_mean.fit_transform(data_np)
    # scale data
    scaler = preprocessing.MinMaxScaler()
    data_norm = pd.DataFrame(scaler.fit_transform(data_np))
    data_norm.columns = data.columns[col:]
    # split into known and unknown
    data_known = data_norm[metadata["Status"].notnull()]
    metadata_known = metadata[metadata["Status"].notnull()]


if __name__ == "__main__":
    main(sys.argv[1:])
