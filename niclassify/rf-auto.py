"""Automatically generate the best possible Random Forest Classifier.

Uses given paramters - quality is limited by number of multirun.
Gives user option to save trained classifier, as well as use previously trained
classifier.
"""

try:
    # import main libraries
    import os
    import logging
    import signal

    import numpy as np
    import pandas as pd
    import seaborn as sns

    # import rest of program modules
    import core

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)


def keyboardInterruptHandler(signal, frame):
    exit(0)


def main():
    """Run the program.

    Takes in user arguments to select data to run on, and outputs a new csv
    with predictions.
    """
    # set up ctrl-c exit
    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    # call some boilerplate functions
    core.boilerplate()

    # get args
    parser, args = core.getargs()

    mode = args.mode

    if mode == "interactive" or mode is None:
        mode,\
            data_file,\
            excel_sheet,\
            selected_cols,\
            class_column,\
            multirun,\
            classifier_file,\
            output_filename,\
            nans = core.interactive_mode()
        data_cols = selected_cols
    else:
        data_file = args.data
        excel_sheet = args.excel_sheet
        selected_cols = args.data_cols
        output_filename = args.out
        nans = args.nanval

        if mode == "train":
            class_column = args.class_column
            multirun = args.multirun
        else:
            classifier_file = args.predict_using

        # get column range for data
        col_range = core.get_col_range(selected_cols)
        data_cols = core.get_data(
            data_file, excel_sheet).columns.values[
                col_range[0]:col_range[1]].tolist()

    # get raw data
    raw_data = core.get_data(data_file, excel_sheet)

    # replace argument-added nans
    if nans is not None:
        raw_data.replace({val: np.nan for val in nans})

    # # remove instances of >1 group count
    # raw_data = raw_data.loc[raw_data["Group_Count"] <= 1]
    # raw_data.reset_index(drop=True, inplace=True)

    # split data into feature data and metadata
    features = raw_data[data_cols]
    metadata = raw_data.drop(data_cols, axis=1)

    # scale data
    feature_norm = core.scale_data(features)

    if mode == "train":  # no classifier provided; train a new one

        # convert class labels to lower if classes are in str format
        if not np.issubdtype(metadata[class_column].dtype, np.number):
            metadata[class_column] = metadata[class_column].str.lower()

        # get only known data and metadata
        features_known, metadata_known = core.get_known(
            feature_norm, metadata, class_column)

        # train classifier
        logging.info("training random forest...")
        forest = core.RandomForestAC().train(
            features_known, metadata_known[class_column], multirun)

        # save confusion matrix
        logging.info("saving confusion matrix...")
        core.save_confm(forest, features_known,
                        metadata_known, class_column, output_filename)

    else:   # classifier provided; load it in
        from joblib import load
        forest = load(classifier_file)

    # impute data
    logging.info("imputing data...")
    feature_norm = core.impute_data(feature_norm)

    # make predictions
    logging.info("predicting unknown class labels...")
    predict = pd.DataFrame(forest.predict(feature_norm))
    # rename predict column
    predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)
    # get predict probabilities
    predict_prob = pd.DataFrame(forest.predict_proba(feature_norm))
    # rename column
    predict_prob.rename(
        columns={
            predict_prob.columns[i]: "prob. {}".format(c)
            for i, c in enumerate(forest.classes_)},
        inplace=True)

    # save output
    logging.info("saving new output...")
    df = pd.concat([metadata, predict, predict_prob, feature_norm], axis=1)
    try:
        output_path = "/".join(output_filename.split("/")[:-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        df.to_csv(output_filename, index=False)
    except (KeyError, FileNotFoundError, OSError):
        logging.error("Output folder creation failed.")
        exit(-1)

    # generate and output graph
    logging.info("generating final graphs...")
    core.output_graph(feature_norm, predict, output_filename)

    logging.info("...done!")

    # if classifier is new, give option to save
    if mode == "train":
        core.save_clf_dialog(forest)


if __name__ == "__main__":
    main()
