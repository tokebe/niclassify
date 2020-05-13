try:
    # import main libraries
    import argparse
    import xlrd

    import pandas as pd
    import seaborn as sns

    # import rest of program modules
    import core

except ModuleNotFoundError:
    print("Missing required modules. Install requirements by running")
    print("'python -m pip install -r requirements.txt'")
    exit(-1)

sns.set()


def main():
    """Run the program.

    Takes in user arguments to select data to run on, and outputs a new csv
    with predictions.
    """
    # get args
    parser = argparse.ArgumentParser()
    args = core.getargs(parser)

    # ensure required folders exist
    core.assure_path()

    # get raw data
    raw_data = core.get_data(parser, args.data, args.excel)

    # get column range for data
    dcol = core.get_col_range(parser, args.dcol)

    # remove instances of >1 group count
    raw_data = raw_data.loc[raw_data["Group_Count"] <= 1]
    raw_data.reset_index(drop=True, inplace=True)

    # split data into feature data and metadata
    data = raw_data.iloc[:, dcol[0]:dcol[1]]
    metadata = raw_data.drop(raw_data.columns[dcol[0]:dcol[1]], axis=1)

    # convert class labels to lower
    metadata.loc[:, args.clabel] = metadata[args.clabel].str.lower()

    # scale data
    data_norm = core.scale_data(data)

    if args.predicton is None:  # no classifier provided; train a new one
        # get only known data and metadata
        data_known, metadata_known = core.get_known(
            data_norm, metadata, args.clabel)

        # get classifier (train it)
        print("training random forest...")
        forest = core.train_forest(data_known, metadata_known,
                                   args.clabel, args.multirun)

        # save confusion matrix
        print("saving confusion matrix...")
        core.save_confm(forest, data_known,
                        metadata_known, args.clabel, args.out)

    else:   # classifier provided; load it in
        from joblib import load
        forest = load(args.predicton)

    # impute data
    print("imputing data...")
    data_norm = core.impute_data(data_norm)

    # make predictions
    predict = pd.DataFrame(forest.predict(data_norm))
    # rename predict column
    predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)
    # get predict probabilities
    predict_prob = pd.DataFrame(forest.predict_proba(data_norm))
    # rename column
    predict_prob.rename(
        columns={predict_prob.columns[0]: "predict prob"}, inplace=True)

    # save output
    print("saving new output...")
    df = pd.concat([metadata, predict, predict_prob, data_norm], axis=1)
    try:
        df.to_csv(args.out, index=False)
    except (KeyError, FileNotFoundError):
        parser.error("intended output folder does not exist!")

    # generate and output graph
    print("generating final graphs...")
    core.output_graph(data_norm, predict, args.out)

    print("...done!")

    # if classifier is new, give option to save
    if args.predicton is None:
        core.save_clf_dialog(forest)


if __name__ == "__main__":
    main()
