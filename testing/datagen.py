import pandas as pd
from sklearn.datasets import make_classification


x, y = make_classification(
    n_samples=1000,
    n_features=100,
    n_classes=2
)

x = pd.DataFrame(x)
y = pd.DataFrame(y)

x_names = {c: "x{}".format(c) for c in x.columns.values.tolist()}

x.rename(columns=x_names, inplace=True)
print(x.head())

y.rename(columns={y.columns[0]: "class"}, inplace=True)
print(y.head())

x = pd.concat([x, y], axis=1)

x.to_csv("test_data_1k_f100.csv", index=False)
