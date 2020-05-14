"""Automatically generate the best possible Random Forest Classifier.

Uses given paramters - quality is limited by number of multirun.
Gives user option to save trained classifier, as well as use previously trained
classifier.
"""

try:
    # import main libraries
    import os
    import logging
    import argparse

    import pandas as pd
    import seaborn as sns

    # import rest of program modules
    import core

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

# set seaborn theme/format
sns.set()

# set log filename
i = 0
while os.path.exists("rf-auto{}.log".format(i)):
    i += 1
logname = "rf-auto{}.log".format(i)

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(logname),
        logging.StreamHandler()
    ]
)


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
    raw_data = core.get_data(parser, args.data, args.excel_sheet)

    # get column range for data
    data_cols = core.get_col_range(parser, args.data_cols)

    # remove instances of >1 group count
    raw_data = raw_data.loc[raw_data["Group_Count"] <= 1]
    raw_data.reset_index(drop=True, inplace=True)

    # split data into feature data and metadata
    data = raw_data.iloc[:, data_cols[0]:data_cols[1]]
    metadata = raw_data.drop(
        raw_data.columns[data_cols[0]:data_cols[1]], axis=1)

    # convert class labels to lower
    metadata.loc[:, args.class_column] = \
        metadata[args.class_column].str.lower()

    # scale data
    data_norm = core.scale_data(data)

    if args.predict_using is None:  # no classifier provided; train a new one
        # get only known data and metadata
        data_known, metadata_known = core.get_known(
            data_norm, metadata, args.class_column)

        # train classifier
        logging.info("training random forest...")
        forest = core.train_forest(data_known, metadata_known,
                                   args.class_column, args.multirun)

        # save confusion matrix
        logging.info("saving confusion matrix...")
        core.save_confm(forest, data_known,
                        metadata_known, args.class_column, args.out)

    else:   # classifier provided; load it in
        from joblib import load
        forest = load(args.predict_using)

    # impute data
    logging.info("imputing data...")
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
    logging.info("saving new output...")
    df = pd.concat([metadata, predict, predict_prob, data_norm], axis=1)
    try:
        df.to_csv(args.out, index=False)
    except (KeyError, FileNotFoundError):
        parser.error("intended output folder does not exist!")

    # generate and output graph
    logging.info("generating final graphs...")
    core.output_graph(data_norm, predict, args.out)

    logging.info("...done!")

    # if classifier is new, give option to save
    if args.predict_using is None:
        core.save_clf_dialog(forest)


if __name__ == "__main__":
    main()
