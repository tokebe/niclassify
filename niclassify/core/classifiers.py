"""Module containing forest training function.

In theory could be modified to contain other classifier trainers.
"""
try:
    import logging

    import numpy as np

    from copy import deepcopy

    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold

    # from itertools import chain, combinations
except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)


class AutoClassifier:
    """A basic template class for an automated classifier.

    See RandomForestAC for an implemented automated classifier.
    """

    def __init__(
            self,
            clf=None,
            hyperparams=None,
            hp_method=None,
            score_method=None):
        self.clf = clf
        self.hyperparams = hyperparams
        self.hp_method = hp_method
        self.score_method = score_method

        self.best_train_predict = None
        self.best_test_predict = None
        self.best_train_score = 0
        self.best_test_score = 0
        self.best_model = None
        self.best_train_cm = None
        self.best_test_cm = None

    # TODO split train's different steps into methods (with args and returns)
    # this will let you implement a special training method with callbacks
    # which can be separate from the normal training method
    # TODO make a generate_report() method which returns the previously logged
    # report as a string, logging can be handled by outside
    # or by another log_report() method
    # TODO decide if any methods should be changed into properties
    def train(
            self,
            data_known,
            metadata_known,
            class_col="Status",
            multirun=1):
        """Train the classifier.

        Automatically selects hyperparameters and generates the best model.

        Args:
            data_known (DataFrame): Known Data.
            metadata_known (DataFram): Known Metadata, including class label.
            class_col (str, optional): Name of metadata column containing class
                labels. Defaults to "Status".
            multirun (int, optional): Number of times to run. Defaults to 1.

        Returns:
            RandomForest: The Trained Random Forest.

        """
        # useful logging
        logging.info("  training using variables:")
        for i in data_known.columns.values.tolist():
            logging.info("    {}".format(i))
        logging.info("  obtaining best hyperparameters...")

        # split data for training
        x_train, x_test, y_train, y_test = train_test_split(
            data_known, metadata_known[class_col],
            stratify=metadata_known[class_col],
            test_size=0.2)  # previously .15

        # get number of k splits based on data available
        k = min(metadata_known[class_col].value_counts())
        # set k to 10 if the minimum class label count is >= 10, or to the min
        # class label count if it's less
        k = 10 if k >= 10 else k

        search = self.hp_method(
            self.cf,
            self.hyperparams,
            n_jobs=-1,
            scoring='balanced_accuracy',
            cv=StratifiedKFold(n_splits=k))

        search.fit(x_train, y_train)

        self.best_model = search.best_estimator_

        self.best_train_predict = self.best_model.predict(x_train)
        self.best_test_predict = self.best_model.predict(x_test)

        self.best_train_score = metrics.balanced_accuracy_score(
            y_train, self.best_train_predict)
        self.best_test_score = metrics.balanced_accuracy_score(
            y_test, self.best_test_predict)

        logging.info("  found best hyperparameters (as follows):")
        for param, val in search.self.best_params_.items():
            logging.info("    {}: {}".format(param, val))

        self.best_train_cm = metrics.confusion_matrix(
            y_train,
            self.best_train_predict,
            labels=metadata_known[class_col].unique(),
            normalize='all'
        )
        self.best_test_cm = metrics.confusion_matrix(
            y_test,
            self.best_test_predict,
            labels=metadata_known[class_col].unique(),
            normalize='all'
        )

        if multirun > 1:
            logging.info(
                "  generating {} classifiers using found parameters...".format(
                    multirun)
            )
            for m in range(multirun):

                x_train, x_test, y_train, y_test = train_test_split(
                    data_known, metadata_known[class_col],
                    stratify=metadata_known[class_col],
                    test_size=0.2)  # previously .15

                print(
                    "    testing forest {} of {}...".format(m + 1, multirun),
                    end="\r")

                model = search.best_estimator_.fit(x_train, y_train)

                train_predict = model.predict(x_train)
                test_predict = model.predict(x_test)

                train_ba = metrics.balanced_accuracy_score(
                    y_train, train_predict)
                test_ba = metrics.balanced_accuracy_score(y_test, test_predict)

                if test_ba > self.best_test_ba:
                    self.best_train_predict = train_predict
                    self.best_test_predict = test_predict
                    self.best_train_ba = train_ba
                    self.best_test_ba = test_ba

                    self.best_train_cm = metrics.confusion_matrix(
                        y_train,
                        train_predict,
                        labels=metadata_known[class_col].unique(),
                        normalize='all'
                    )
                    self.best_test_cm = metrics.confusion_matrix(
                        y_test,
                        test_predict,
                        labels=metadata_known[class_col].unique(),
                        normalize='all'
                    )

                    self.best_model = deepcopy(model)
                    # deep copy ensures best model is saved, otherwise an alias
                    # would be saved and overwritten on next fit\

        if multirun > 1:
            print("\n")

        logging.info("out-of-bag score: {}".format(self.best_model.oob_score_))
        logging.info("---")

        logging.info("train set BA: {}".format(self.best_train_ba))
        labels = metadata_known[class_col].unique()
        # logging.info(self.best_train_cm)
        for true, pred in np.ndindex(self.best_train_cm.shape):
            if true == pred:
                continue
            else:
                logging.info(
                    "train set: percent of {} mislabeled to {}: {:.2f}%".format(
                        labels[true],
                        labels[pred],
                        (self.best_train_cm[true, pred] * 100)))
        logging.info("---")
        logging.info("test set BA : {}".format(self.best_test_ba))
        # logging.info(self.best_test_cm)
        for true, pred in np.ndindex(self.best_test_cm.shape):
            if true == pred:
                continue
            else:
                logging.info(
                    "test set: percent of {} mislabeled to {}: {:.2f}%".format(
                        labels[true],
                        labels[pred],
                        (self.best_test_cm[true, pred] * 100)))
        logging.info("---")

        return self.best_model

    print()


class RandomForestAC(AutoClassifier):
    """A prefab Random Forest AutoClassifier, no additional setup required.

    New AutClassifiers should be  able to be defined in the same manner without
    needing to overwrite train().
    """

    def __init__(self):
        """Initialize the Random Forest Auto Classifier.

        Technically this could be achieved without making a new subclass,
        however this makes things easier for repeated use.
        """
        super().__init__(
            clf=RandomForestClassifier(
                class_weight="balanced", oob_score=True, n_estimators=1000
            ),
            hyperparams={
                'max_depth': list(np.linspace(10, 50).astype(int)),
                'min_samples_split': list(np.arange(0.05, 0.5, 0.05))
            },
            hp_method=RandomizedSearchCV,
            score_method='balanced_accuracy',
        )
