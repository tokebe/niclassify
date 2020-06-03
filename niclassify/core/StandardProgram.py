

try:
    # import main libraries
    import os
    import logging
    import signal

    import numpy as np
    import pandas as pd
    import seaborn as sns

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

# import rest of program modules
from . import utilities
from . import classifiers

# TODO fix folder references when running through __main__.py and rf_auto.py
# TODO fix logging not logging anything
# checked: logs properly when running from rf_auto.py
# TODO fix classifier save not saving anything
# saves properly when running from rf_auto.py
# TODO see what that runtimewarning is about
# TODO basic bugfixing
# TODO add documentation
# TODO add a bunch of user error checking both here and in utilities


class StandardProgram:
    def __init__(self, clf, arg_parser=None, interactive_parser=None):
        self.clf = clf
        self.arg_parser = arg_parser
        self.interactive_parser = interactive_parser

        self.mode = None
        self.data_file = None
        self.excel_sheet = None
        self.selected_cols = None
        self.data_cols = None
        self.class_column = None
        self.multirun = None
        self.classifier_file = None
        self.output_filename = None
        self.nans = None

    def boilerplate(self):
        # set seaborn theme/format
        sns.set()

        # ensure required folders exist
        utilities.assure_path()

        # set log filename
        i = 0
        while os.path.exists("../output/logs/rf-auto{}.log".format(i)):
            i += 1
        logname = "../output/logs/rf-auto{}.log".format(i)

        # set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(logname),
                logging.StreamHandler()
            ]
        )

    def get_args(self):
        parser, args = self.arg_parser()

        self.mode = args.mode

        if ((self.mode == "interactive" or self.mode is None)
                and self.interactive_parser is None):
            raise ValueError(
                "Mode is interactive but interactive parser is not provided!")

        elif ((self.mode == "interactive" or self.mode is None)
                and self.interactive_parser is not None):
            self.mode,\
                self.data_file,\
                self.excel_sheet,\
                self.selected_cols,\
                self.class_column,\
                self.multirun,\
                self.classifier_file,\
                self.output_filename,\
                self.nans = self.interactive_parser()
            self.data_cols = self.selected_cols

        else:
            self.data_file = args.data
            self.excel_sheet = args.excel_sheet
            self.selected_cols = args.data_cols
            self.output_filename = args.out
            self.nans = args.nanval

            if self.mode == "train":
                self.class_column = args.class_column
                self.multirun = args.multirun
            else:
                self.classifier_file = args.predict_using

            # get column range for data
            self.col_range = utilities.get_col_range(self.selected_cols)
            self.data_cols = utilities.get_data(
                self.data_file,
                self.excel_sheet
            ).columns.values[self.col_range[0]:self.col_range[1]].tolist()

    def prep_data(self):
        # get raw data
        raw_data = utilities.get_data(self.data_file, self.excel_sheet)

        # replace argument-added nans
        if self.nans is not None:
            raw_data.replace({val: np.nan for val in self.nans})

        # split data into feature data and metadata
        features = raw_data[self.data_cols]
        metadata = raw_data.drop(self.data_cols, axis=1)

        # scale data
        feature_norm = utilities.scale_data(features)

        return raw_data, features, feature_norm, metadata

    def train_AC(self, feature_norm, metadata):
        # convert class labels to lower if classes are in str format
        if not np.issubdtype(
                metadata[self.class_column].dtype, np.number):
            metadata[self.class_column] = \
                metadata[self.class_column].str.lower()

        # get only known data and metadata
        features_known, metadata_known = utilities.get_known(
            feature_norm, metadata, self.class_column)

        # train classifier
        logging.info("training random forest...")
        clf = classifiers.RandomForestAC().train(
            features_known, metadata_known[self.class_column], self.multirun)

        # save confusion matrix
        logging.info("saving confusion matrix...")
        utilities.save_confm(
            clf,
            features_known,
            metadata_known,
            self.class_column,
            self.output_filename
        )

        return clf

    def load_classifier(self):
        from joblib import load
        classifier = load(self.classifier_file)

        return classifier

    def predict_AC(self, clf, feature_norm):
        # impute data
        logging.info("imputing data...")
        feature_norm = utilities.impute_data(feature_norm)

        # make predictions
        logging.info("predicting unknown class labels...")
        predict = pd.DataFrame(clf.predict(feature_norm))
        # rename predict column
        predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)

        # check if classifier supports proba and use it if so
        proba_method = getattr(clf, "predict_proba", None)
        if proba_method is not None and callable(proba_method):

            # get predict probabilities
            predict_prob = pd.DataFrame(clf.predict_proba(feature_norm))
            # rename column
            predict_prob.rename(
                columns={
                    predict_prob.columns[i]: "prob. {}".format(c)
                    for i, c in enumerate(clf.classes_)},
                inplace=True)

            return predict, predict_prob

        else:
            return (predict)

    def save_outputs(
            self,
            clf,
            feature_norm,
            metadata,
            predict,
            predict_prob=None
    ):
        # save predictions
        utilities.save_predictions(
            metadata,
            predict,
            feature_norm,
            self.output_filename,
            predict_prob
        )

        # generate and output graph
        logging.info("generating final graphs...")
        utilities.output_graph(feature_norm, predict, self.output_filename)

        logging.info("...done!")

        # if classifier is new, give option to save
        if self.mode == "train":
            utilities.save_clf_dialog(clf)

    def default_run(self):
        # set up the boilerplate
        self.boilerplate()
        # get required arguments to run the program
        self.get_args()
        # process the data for use
        raw_data, features, feature_norm, metadata = self.prep_data()

        # if the mode is train, train the classifier
        if self.mode == "train":
            clf = self.train_AC(feature_norm, metadata)
        else:
            clf = self.load_classifier()

        # By default, predictions are made whether the mode was train or not.
        predict = self.predict_AC(clf, feature_norm)
        # Because predict_AC may return predict_prob we check if it's a tuple
        # and act accordingly
        if type(predict) == tuple:
            predict, predict_prob = predict
        else:
            predict_prob = None

        self.save_outputs(clf, feature_norm, metadata, predict, predict_prob)
