import pandas as pd
from memory_profiler import memory_usage


@profile
def over(df):
    df = df[df.notnull()]
    print(df)


@profile
def cpy(df):
    ndf = df[df.notnull()]
    print(df)
    print(ndf)


@profile
def part(df):
    print(df)
    print(df["green"])


@profile
def cat(df):
    ndf = df.copy()
    nndf = pd.concat([df, ndf], axis=1)
    print(nndf)


if __name__ == "__main__":

    data = {
        "red": [1, None, 3, 10],
        "green": [4, 5, 6, 11],
        "blue": [7, 8, None, 12]
    }

    df = pd.DataFrame.from_dict(data)

    print(df)

    cpy(df)
    over(df)
    part(df)
    cat(df)


# run this with python -m memory_profiler ramtest.py
