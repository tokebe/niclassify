"""
StandardProgram class and not much else.

StandardProgram may be used as a helper for creating new programs based upon
it, as seen in the GUI implementation.
"""

try:
    # import main libraries
    import os
    import logging
    import tempfile

    import numpy as np
    import pandas as pd
    import seaborn as sns

    from functools import partial
    from multiprocessing import Pool
    from multiprocessing import cpu_count
    from requests.exceptions import RequestException
    from time import sleep

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)

# import rest of program modules
from . import utilities
from . import classifiers


def parallelize(df, func, n_cores=None):
    """
    Parallelize applying a function to a DataFrame.

    Args:
        df (DataFrame): DataFrame to apply function to.
        func (func): A function suitable for df.apply().
        n_cores (int, optional): Number of processes to use. Defaults to None.

    Returns:
        df: The resulting output.

    """
    if n_cores is None:
        n_cores = cpu_count()

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def run_on_subset(func, data_subset):
    """
    Apply a function to a subset of a DataFrame.

    Args:
        func (func): A function suitable for df.apply().
        data_subset (DataFrame): A subset of a larger DataFrame.

    Returns:
        DataFrame: The resulting output subset.

    """
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, n_cores=None):
    """
    Parallelize applying a function to a DataFrame, row-wise.

    Args:
        data (DataFrame): A DataFrame.
        func (func): A function suitable for df.apply().
        n_cores (int, optional): Number of processes to use. Defaults to None.

    Returns:
        DataFrame: The resulting output.

    """
    return parallelize(data, partial(run_on_subset, func), n_cores)


class StandardProgram:
    """
    A standard template for the classifier program.

    Contains all methods required to run the program, using either default_run,
    or a user-made method/override. Most methods handle some basic data
    manipulation to prepare data for use in utility methods, defined in
    utilities.
    """

    def __init__(self, clf, arg_parser=None, interactive_parser=None):
        """
        Instantiate the program.

        Args:
            clf (AutoClassifier): An AutoClassifier such as RandomForestAC.
            arg_parser (function, optional): A function which returns a number
                of items (see get_args). Defaults to None.
            interactive_parser (function, optional): A function which returns a
                number of items (see get_args), preferably getting them
                interactively. Defaults to None.
        """
        self.clf = clf
        self.arg_parser = arg_parser
        self.interactive_parser = interactive_parser

        self.mode = None

        # stored variables for data retrieval program
        self.geo = None
        self.taxon = None
        self.api = None
        self.ref_geo = None
        self.request_fname = None
        self.filtered_fname = None
        self.fasta_fname = None
        self.fasta_align_fname = None
        self.delim_fname = None
        self.tree_fname = None
        self.seq_features_fname = None
        self.finalized_fname = None

        # stored variables for classification program
        self.data_file = None
        self.excel_sheet = None
        self.feature_cols = None
        self.class_column = None
        self.multirun = None
        self.classifier_file = None
        self.output_filename = None
        self.nans = None

    def align_fasta(self, external=False):
        """
        Align the fasta file.

        Args:
            external (bool, optional): Open in an external prompt
                (non-blocking). Defaults to False.
        """
        utilities.align_fasta(
            self.fasta_fname, self.fasta_align_fname, external)

        if os.stat(self.fasta_align_fname).st_size == 0:
            raise ChildProcessError("Sequence Alignment Failed")

    def boilerplate(self):
        """
        Set up the theme, ensure required folders exist, and set up logging.

        Consider overriding if you need additional preparations for every run.

        Returns:
            str: the path to the log file to be written to

        """
        # set seaborn theme/format
        sns.set()

        # ensure required folders exist
        utilities.assure_path()

        # # set log filename
        # i = 0
        # while os.path.exists(
        #     os.path.join(
        #         utilities.MAIN_PATH,
        #         "output/logs/rf-auto{}.log".format(i)
        #     )
        # ):
        #     i += 1
        # logname = os.path.join(
        #     utilities.MAIN_PATH,
        #     "output/logs/rf-auto{}.log".format(i)
        # )
        # set log filename (overwriting previous)
        logname = os.path.join(
            utilities.MAIN_PATH,
            "output/logs/rf-auto{}.log".format(0)
        )

        # set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(logname),
                logging.StreamHandler()
            ]
        )

        return logname

    def check_file_exists(self, filename):
        """
        Check if the given filename exists in the system directory.

        Args:
            filename (str): A filename.

        Raises:
            ValueError: If the file does not exist.

        Returns:
            Bool: True if file exists.

        """
        # convert filename to proper path
        if not os.path.isabs(filename):
            filename = os.path.join(utilities.MAIN_PATH, filename)

        # check if filename exists
        if not os.path.exists(filename):
            raise ValueError("file {} does not exist.".format(filename))
            return False  # this line likely unreachable.

        return True

    def check_native_gbif(self, row):
        """
        Check if a given species is native according to gbif.

        Args:
            row (df row): A DataFrame row with a column "species_name".

        Returns:
            str or np.NaN: Native, Introduced, or np.NaN if unknown.

        """
        if pd.isna(row["species_name"]):
            return np.NaN
        elif len(row["species_name"]) == 0:
            return np.NaN

        # check GBIF native ranges
        try:
            nranges = utilities.get_native_ranges(row["species_name"])
        except RequestException:
            return np.NaN

        # no rate limit provided, so we'll just wait a bit to be safe
        sleep(0.01)

        if nranges is None:
            return np.NaN

        crypto = False

        for nrange in nranges:
            if nrange == "Cosmopolitan, Cryptogenic":
                print("  {}: cryptogenic".format(row["species_name"]))
                crypto = True
                continue
            if nrange == "Pantropical, Circumtropical":
                print("  {}:".format(row["species_name"]))
                ref_hierarchy = utilities.get_ref_hierarchy(self.ref_geo)
                if ref_hierarchy["Pantropical"] is True:
                    return "Native"
                else:
                    continue
            if nrange == "Subtropics":
                print("  {}: subtropics".format(row["species_name"]))
                ref_hierarchy = utilities.get_ref_hierarchy(self.ref_geo)
                if ref_hierarchy["Subtropics"] is True:
                    return "Native"
                else:
                    continue

            if nrange == self.ref_geo:
                print("  {}: direct native to ref".format(row["species_name"]))
                return "Native"
            elif (utilities.geo_contains(self.ref_geo, nrange)
                  or utilities.geo_contains(nrange, self.ref_geo)):
                print("  {}: {} <=> {}".format(
                    row["species_name"], self.ref_geo, nrange))
                return "Native"
            else:
                print("  {}: {} <!=> {}".format(
                    row["species_name"], self.ref_geo, nrange))

        # if it hasn't found a reason to call it native
        return "Introduced" if not crypto else np.NaN

    def check_native_itis(self, row):
        """
        Check if a given species is native according to itis.

        Args:
            row (df row): A DataFrame row with a column "species_name".

        Returns:
            str or np.NaN: Native, Introduced, or np.NaN if unknown.

        """
        if pd.isna(row["species_name"]):
            return np.NaN
        elif len(row["species_name"]) == 0:
            return np.NaN

        # check ITIS jurisdictions
        try:
            jurisdictions = utilities.get_jurisdictions(row["species_name"])
        except RequestException:
            return np.NaN

        # no rate limit provided, so we'll just wait a bit to be safe
        sleep(0.01)

        if jurisdictions is None:
            print("  no jurisdiction(s) returned")
            return np.NaN

        # simple first: check if jurisdiction is just current reference geo
        for jurisdiction, status in jurisdictions.items():
            print("{}, {}".format(jurisdiction, status))
            if jurisdiction == self.ref_geo:
                return status if status != "Native&Introduced" else np.NaN
            # otherwise if one contains the other it's native
            elif (utilities.geo_contains(self.ref_geo, jurisdiction)
                  or utilities.geo_contains(jurisdiction, self.ref_geo)):
                print("{}: {} <=> {}".format(
                    row["species_name"], self.ref_geo, jurisdiction))
                return status if status != "Native&Introduced" else np.NaN
            else:
                print("{}: {} <!=> {}".format(
                    row["species_name"], self.ref_geo, jurisdiction))

        # if it hasn't found a reason to call it native
        return "Introduced"  # this maybe should be NA

    def delimit_species(self, method="GMYC", external=False):
        """
        Delimit species by their nucleotide sequences.

        Args:
            method (str, optional): Delimitation method. Defaults to "GMYC".
            external (bool, optional): Run the delimitation concurrently in an
                external console. Defaults to None.
        """
        if method == "GMYC":
            delimit = utilities.delimit_species_GMYC
        elif method == "bPTP":
            delimit = utilities.delimit_species_bPTP
        else:
            raise KeyError("Specified delimitation method does not exist.")

        delimit(
            self.fasta_align_fname,
            self.tree_fname,
            self.delim_fname,
            external
        )

        if os.stat(self.tree_fname).st_size == 0:
            raise ChildProcessError("Species delimitation failed.")

        if os.stat(self.delim_fname).st_size == 0:
            raise ChildProcessError("Species delimitation failed.")

        # merging in delimitation is now handled after feature generation
        # read in filtered sequence data and new delim file for combining
        # seq_filtered = utilities.get_data(self.filtered_fname)

        # delim = utilities.get_data(self.delim_fname)

        # # rename sample_name to processid for merge
        # delim.rename(columns={"sample_name": "processid"}, inplace=True)

        # # left join gymc species back into filtered data
        # seq_filtered = pd.merge(
        #     seq_filtered, delim, on="processid", how="left")

        # seq_filtered.to_csv(self.finalized_fname,  index=False)

    def generate_features(self, external=False):
        utilities.generate_measures(
            self.fasta_align_fname,
            self.delim_fname,
            self.seq_features_fname,
            external=external
        )

        if os.stat(self.seq_features_fname).st_size == 0:
            raise ChildProcessError("Feature generation failed.")

        features = utilities.get_data(self.seq_features_fname)
        meta = utilities.get_data(self.filtered_fname)

        meta = pd.merge(
            meta, features, on="processid", how="left")

        meta.to_csv(self.finalized_fname, index=False)

    def get_args(self):
        """
        Get arguments from either parser and store required values.

        Raises:
            ValueError: If interactive parser is expected by parser but not
                provided.

        """
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
                self.feature_cols,\
                self.class_column,\
                self.multirun,\
                self.classifier_file,\
                self.output_filename,\
                self.nans = self.interactive_parser()

        else:
            self.data_file = args.data
            self.excel_sheet = args.excel_sheet
            self.feature_cols = args.feature_cols
            self.output_filename = args.out
            self.nans = args.nanval

            if self.mode == "train":
                self.class_column = args.class_column
                self.multirun = args.multirun
            else:
                self.classifier_file = args.pfeaturesct_using

            # if feature cols is a range str, convert to list of names
            if type(self.feature_cols) is str:
                self.check_file_exists("data/" + self.data_file)
                col_range = utilities.get_col_range(self.selected_cols)
                self.feature_cols = utilities.get_data(
                    self.data_file,
                    self.excel_sheet
                ).columns.values[col_range[0]:col_range[1]].tolist()

        # do some error checking
        self.check_file_exists("data/" + self.data_file)
        if self.classifier_file is not None:
            self.check_file_exists(
                "output/classifiers/" + self.classifier_file)

    def get_sequence_data(self):
        """
        Read in unprepared sequence data and return it.

        Returns:
            DataFrame: DataFrame of sequence data.

        """
        return utilities.get_data(self.request_fname)

    def impute_data(self, feature_norm):
        """
        Impute given data.

        Mostly a wrapper for utilities.impute_data().

        Args:
            feature_norm (DataFrame): Feature data in DataFrame.

        Returns:
            DataFrame: The imputed feature data.

        """
        # type error checking
        if type(feature_norm) is not pd.DataFrame:
            raise TypeError("Cannot impute: feature_norm is not DataFrame.")
        # impute data
        logging.info("imputing data...")
        return utilities.impute_data(feature_norm)

    def lookup_status(self):
        """Look up the statuses of species on GBIF and ITIS."""
        # look up the status through ITIS and GBIF and add native/introduced
        # columns

        # get data
        data = utilities.get_data(self.finalized_fname)

        # make a table of all unique species
        # this should somewhat reduce the number of lookups
        species = pd.DataFrame({"species_name": data["species_name"].unique()})

        # lookup statuses and create columns
        print("Getting gbif statuses...")
        # data["gbif_status"] = data.apply(self.check_native_gbif, axis=1)
        species["gbif_status"] = parallelize_on_rows(
            species, self.check_native_gbif)

        # print("Getting itis statuses...")
        # data["itis_status"] = data.apply(self.check_native_itis, axis=1)
        species["itis_status"] = parallelize_on_rows(
            species, self.check_native_itis)

        def combine_status(row):
            """Combine given GBIF and ITIS statuses."""
            if row["itis_status"] == row["gbif_status"]:
                return row["itis_status"]
            elif (pd.isnull(row["itis_status"])
                  and pd.notnull(row["gbif_status"])):
                return row["gbif_status"]
            elif (pd.isnull(row["gbif_status"])
                  and pd.notnull(row["itis_status"])):
                return row["itis_status"]
            else:
                return np.NaN

        print("combining statuses...")
        # create a column based on agreement of previous two
        species["final_status"] = species.apply(combine_status, axis=1)

        # merge species table back in
        data = pd.merge(data, species, on="species_name", how="left")

        # print("saving statuses...")
        # save changes
        data.to_csv(self.finalized_fname, index=False)

    def predict_AC(self, clf, feature_norm, status_cb=None):
        """
        Predict using a trained AutoClassifier and return predictions.

        Args:
            clf (AutoClassifier): A Trained AutoClassifier.
            feature_norm (DataFrame): Normalized, imputed feature data.
            status_cb(func, optional): Callback function to update status.
                Defaults to None.

        Returns:
            DataFrame or tuple: Predictions, and probabilities if supported
                by AC.

        """
        # type error checking
        if type(feature_norm) is not pd.DataFrame:
            raise TypeError("Cannot predict: feature_norm is not DataFrame.")
        if not isinstance(clf, classifiers.AutoClassifier):
            raise TypeError("Cannot predict: classifier does not inherit from \
AutoClassifier")

        # prep for a few tests
        # feature names are case-insenstive
        clf_expect = [x.lower() for x in clf.trained_features]
        ft_given = [x.lower() for x in feature_norm.columns.values.tolist()]

        # ensure same number of features between data and classifier
        if len(clf_expect) != len(ft_given):
            raise ValueError(
                "Classifier expects different number of features than provided"
            )
        # ensure that feature columns match between data and trained classifier
        elif set(clf_expect) != set(ft_given):
            raise KeyError(
                "Given feature names do not match those expected by classifier"
            )

        status_cb("Normalizing features...")

        # sort features as classifier expects so vals are considered correctly
        feature_norm = feature_norm[clf.trained_features]

        status_cb("Making predictions...")

        # make predictions
        logging.info("predicting unknown class labels...")
        predict = pd.DataFrame(clf.clf.predict(feature_norm))
        # rename predict column
        predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)

        # check if classifier supports proba and use it if so
        proba_method = getattr(clf.clf, "predict_proba", None)
        if proba_method is not None and callable(proba_method):

            status_cb("Getting predicition probabilities...")

            # get predict probabilities
            predict_prob = pd.DataFrame(clf.clf.predict_proba(feature_norm))
            # rename column
            predict_prob.rename(
                columns={
                    predict_prob.columns[i]: "prob. {}".format(c)
                    for i, c in enumerate(clf.clf.classes_)
                },
                inplace=True)

            return predict, predict_prob

        else:
            return (predict)

    def prep_data(self):
        """
        Get and prepare data for use.

        Returns:
            tuple: Of raw data, feature data, normalized feature data, and
                metadata.

        """
        # get raw data
        metadata = utilities.get_data(self.data_file, self.excel_sheet)

        # replace argument-added nans
        if self.nans is not None:
            metadata.replace({val: np.nan for val in self.nans}, inplace=True)

        # split data into feature data and metadata
        features = metadata[self.feature_cols]
        metadata = metadata.drop(self.feature_cols, axis=1)

        # convert class labels to lower if classes are in str format
        if self.class_column is not None:
            if not np.issubdtype(
                    metadata[self.class_column].dtype, np.number):
                metadata[self.class_column] = \
                    metadata[self.class_column].str.lower()

        # scale (normalize) data
        features = utilities.scale_data(features)

        return features, metadata

    def prep_sequence_data(self, data):
        """
        Prepare sequence data.

        Args:
            data (DataFrame): DataFrame of sequence data.

        Returns:
            DataFrame: Prepared sequence data.

        """
        data = utilities.prep_sequence_data(data)
        utilities.write_fasta(data, self.fasta_fname)
        return data

    def print_vars(self):
        """
        Print all the currently stored vars.

        Basically just here for debugging.
        """
        print("mode: {}".format(self.mode))
        print("data_file: {}".format(self.data_file))
        print("excel_sheet: {}".format(self.excel_sheet))
        print("feature_cols: {}".format(self.feature_cols))
        print("class_column: {}".format(self.class_column))
        print("multirun: {}".format(self.multirun))
        print("classifier_file: {}".format(self.classifier_file))
        print("output_filename: {}".format(self.output_filename))
        print("nans: {}".format(self.nans))

    def retrieve_sequence_data(self):
        """Retrieve sequence data from api, saving it to filename."""
        utilities.get_geo_taxon(
            self.request_fname, self.geo, self.taxon, self.api)

    def save_outputs(
            self,
            clf,
            features,
            metadata,
            predict,
            predict_prob=None
    ):
        """
        Save the outputs of a prediction.

        Includes data with new predictions, a pairplot, and the classifier if
            it is newly trained.

        Args:
            clf (AutoClassifier): A trained Autoclassifier
            features (DataFrame): Normalized, imputed feature data.
            metadata (DataFrame): Any metadata the original data file contained
            predict (DataFrame): Predicted class labels.
            predict_prob (DataFrame, optional): Prediction probabilities, if
                supported by AC. Defaults to None.
        """
        # type error checking
        if not isinstance(clf, classifiers.AutoClassifier):
            raise TypeError("Cannot save: classifier does not inherit from \
                AutoClassifier")
        if type(features) is not pd.DataFrame:
            raise TypeError("Cannot save: features is not DataFrame.")
        if type(metadata) is not pd.DataFrame:
            raise TypeError("Cannot save: metadata is not DataFrame.")
        if type(predict) is not pd.DataFrame:
            raise TypeError("Cannot save: predict is not DataFrame.")
        if predict_prob is not None and type(predict_prob) is not pd.DataFrame:
            raise TypeError("Cannot save: predict_prob is not DataFrame.")

        # get only known data and metadata
        features_known, metadata_known = utilities.get_known(
            features, metadata, self.class_column)

        # save confusion matrix
        logging.info("saving confusion matrix...")
        utilities.save_confm(
            clf,
            features_known,
            metadata_known[self.class_column],
            self.output_filename
        )

        # save predictions
        utilities.save_predictions(
            metadata,
            predict,
            features,
            self.output_filename,
            predict_prob
        )

        # generate and output graph
        logging.info("generating final graphs...")
        utilities.save_pairplot(features, predict, self.output_filename)

        logging.info("...done!")

        # if classifier is new, give option to save
        if self.mode == "train":
            utilities.save_clf_dialog(clf)

    def train_AC(self, features, metadata, status_cb=None):
        """
        Train the AutoClassifier.

        Prepares data for training.

        Args:
            features (DataFrame): Normalized feature data, preferably with
                nan values removed.
            metadata (DataFrame): Metadata from original data file, including
                known class column.
            status_cb(func, optional): Callback function to update status.
                Defaults to None.

        Returns:
            AutoClassifier: The trained AutoClassifier

        """
        # type error checking

        if type(features) is not pd.DataFrame:
            raise TypeError("Cannot save: features is not DataFrame.")
        if type(metadata) is not pd.DataFrame:
            raise TypeError("Cannot save: metadata is not DataFrame.")

        if status_cb is not None:
            status_cb("Getting data...")

        # get only known data and metadata
        features, metadata = utilities.get_known(
            features, metadata, self.class_column)

        # train classifier
        logging.info("training random forest...")

        if status_cb is not None:
            status_cb("Training random forest...")

        self.clf.train(
            features, metadata[self.class_column], self.multirun, status_cb)

        return self.clf

    def default_run(self):
        """
        Run the program with default methods and settings.

        Generally, you don't need to override this, but instead want to
            override other methods in DefaultProgram, unless you have some very
            specific steps to change.
        """
        # set up the boilerplate
        self.boilerplate()
        # get required arguments to run the program
        self.get_args()
        # process the data for use
        features, metadata = self.prep_data()

        # if the mode is train, train the classifier
        if self.mode == "train":
            clf = self.train_AC(features, metadata)
        else:
            clf = utilities.load_classifier(self.classifier_file)

        # impute the data
        features = self.impute_data(features)

        # By default, predictions are made whether the mode was train or not.
        predict = self.predict_AC(clf, features)
        # Because predict_AC may return predict_prob we check if it's a tuple
        # and act accordingly
        if type(predict) == tuple:
            predict, predict_prob = predict
        else:
            predict_prob = None

        self.save_outputs(clf, features, metadata, predict, predict_prob)
