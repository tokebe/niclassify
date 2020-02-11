import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


def main():

    res = pd.read_excel(
        open("data/Copy of HymenopteraSequenceDataNE.xlsx", "rb"),
        sheet_name="Results",
        usecols="A:F"
    )
    print(res.head())
    print(res.dtypes)


if __name__ == "__main__":
    main()
