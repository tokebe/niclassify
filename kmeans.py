import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans


def main():

    res = pd.read_excel(
        open("data/Copy of HymenopteraSequenceDataNE.xlsx", "rb"),
        sheet_name="Results",
        usecols="A:F"
    )

    res.head()

    res["Status"] = res["Status"].replace({"unknown": np.nan})

    set(res["Status"])

    res["Similarity"] = res["Similarity"].fillna(res["Similarity"].mean())

    x = res.iloc[:, 2:4].values
    scaler = preprocessing.MinMaxScaler()
    resnorm = pd.DataFrame(scaler.fit_transform(x))
    resnorm = pd.concat([res.iloc[:, 0:2], resnorm, res.iloc[:, 4:]], axis=1)

    known = res[res["Status"].notnull()]


def create_kmeans(x, k):
    """Create list of cluster ID's using k-means clustering.

    Args:
        x (ndarray): Data to cluster.
        k (int): number of clusters to make

    Returns:
        array: cluster ID's

    """
    kmeans = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=k)
    )

    return kmeans.fit_predict(x)
