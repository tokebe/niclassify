"""
Module containing AutoClassifier class and a prefab subclass.

AutoClassifier may be used to create new auto-training classifiers with
relative ease.
"""
try:
    import logging

    import numpy as np

    from copy import deepcopy
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

except ModuleNotFoundError:
    logging.error("Missing required modules. Install requirements by running")
    logging.error("'python -m pip install -r requirements.txt'")
    exit(-1)


class AutoClassifier:
    """
    A basic template class for an automated classifier.

    See RandomForestAC for an implemented automated classifier.
    """

    def __init__(
        self,
        clf=None,
        hyperparams=None,
        hp_method=None,
        score_method=None
    ):
        """
        Instantiate the AutoClassifier.

        Args:
            clf (Classifier, optional): An SKlearn Classifier.
                Defaults to None.
            hyperparams (dict, optional): Dictionary of parameters and values
                to test. Defaults to None.
            hp_method (Method, optional): Either GridSearchCV or
                RandomizedSearchCV. Defaults to None.
            score_method (Method, optional): Scoring method to use during
                training. Should be a metric, not a scorer. Defaults to None.
        """
        self.clf = clf
        self.hyperparams = hyperparams
        self.hp_method = hp_method
        self.score_method = score_method

        self.labels = None

        self.best_x_train = None
        self.best_x_test = None
        self.best_y_train = None
        self.best_y_test = None

        self.best_train_predict = None
        self.best_test_predict = None
        self.best_train_score = 0
        self.best_test_score = 0
        self.best_model = None
        self.best_train_cm = None
        self.best_test_cm = None

    def _get_k(self, classes_known):
        """
        Get the k number of splits for a stratified k-fold.

        Args:
            classes_known (Series): The known classes for training on.

        Returns:
            int: The number of splits for the cross-validation. See below.

        """
        # get number of k splits based on data available
        k = min(classes_known.value_counts())
        # set k to 10 if the minimum class label count is >= 10, or to the min
        # class label count if it's less
        k = 10 if k >= 10 else k

        return k

    def _get_model_mean_score(self, model, x_train, x_test, y_train, y_test):
        """
        Get the mean score of the trained model.

        Args:
            model (Classifier): The trained classifier model.
            x_train (np.ndarray): Feature data used in training.
            x_test (np.ndarray): Feature data to used for testing.
            y_train (np.array): Known class labels for training.
            y_test (np.array): Known class labels for testing.

        Returns:
            float: The mean of training and testing scores for the model.

        """
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        train_score = self.score_method(
            y_train, train_predict)
        test_score = self.score_method(
            y_test, test_predict)

        return np.mean((train_score, test_score))

    def _get_test_train_splits(self, features_known, classes_known):
        """
        Split the data into training and testing segments.

        Args:
            features_known (DataFrame): Feature data for known cases.
            classes_known (Series): Known class labels for given feature data.
        """
        # split data for training
        x_train, x_test, y_train, y_test = train_test_split(
            features_known, classes_known,
            stratify=classes_known,
            test_size=0.2)  # previously .15

        return x_train, x_test, y_train, y_test

    def _set_bests(
            self,
            x_train,
            x_test,
            y_train,
            y_test,
            classes_known,

    ):
        """
        Save the relevant data on the best model.

        Args:
            x_train (np.ndarray): Feature data used in training.
            x_test (np.ndarray): Feature data to used for testing.
            y_train (np.array): Known class labels for training.
            y_test (np.array): Known class labels for testing.
            classes_known (Series): Known class labels for given feature data.
        """
        self.best_x_train = x_train
        self.best_x_test = x_test
        self.best_y_train = y_train
        self.best_y_test = y_test

        self.best_train_predict = self.best_model.predict(x_train)
        self.best_test_predict = self.best_model.predict(x_test)

        self.best_train_score = self.score_method(
            y_train, self.best_train_predict)
        self.best_test_score = self.score_method(
            y_test, self.best_test_predict)

        self.best_train_cm = metrics.confusion_matrix(
            y_train,
            self.best_train_predict,
            labels=self.labels,
            normalize='all'
        )
        self.best_test_cm = metrics.confusion_matrix(
            y_test,
            self.best_test_predict,
            labels=self.labels,
            normalize='all'
        )

    def get_report(self):
        """
        Return a report of the best model's performance as a string.

        Returns:
            str: The report.

        """
        report = []

        report.append(
            "out-of-bag score: {}".format(self.best_model.oob_score_))
        report.append("---")

        report.append(
            "train/test scoring method: {}".format(self.score_method.__name__))
        report.append("---")

        report.append("train set score: {}".format(self.best_train_score))
        labels = self.labels
        # report.append(self.best_train_cm)
        for true, pred in np.ndindex(self.best_train_cm.shape):
            if true == pred:
                continue
            else:
                report.append(
                    "train set: percent of {} mislabeled to {}: {:.2f}%".format(
                        labels[true],
                        labels[pred],
                        (self.best_train_cm[true, pred] * 100))
                )
        report.append("---")
        report.append("test set score: {}".format(self.best_test_score))
        # report.append(self.best_test_cm)
        for true, pred in np.ndindex(self.best_test_cm.shape):
            if true == pred:
                continue
            else:
                report.append(
                    "test set: percent of {} mislabeled to {}: {:.2f}%".format(
                        labels[true],
                        labels[pred],
                        (self.best_test_cm[true, pred] * 100)))
        report.append("---")

        return "\n".join(report)

    def log_report(self):
        """
        Get the model performance report and log it.

        Assumes that appropriate logging has been set up already.
        """
        logging.info(self.get_report())

    def train(
            self,
            features_known,
            classes_known,
            multirun=1
    ):
        """
        Train the classifier.

        Automatically selects hyperparameters and generates the best model.

        Args:
            features_known(DataFrame): Features for known classes.
            classes_known(Series): Known classes for given features.
            multirun(int, optional): Number of times to run. Defaults to 1.

        Returns:
            Classifier: The trained classifier.

        """
        # useful logging
        logging.info("  training {}".format(self.clf.__class__.__name__))
        logging.info("  using variables:")
        for i in features_known.columns.values.tolist():
            logging.info("    {}".format(i))
        logging.info("  obtaining best hyperparameters...")

        self.labels = classes_known.unique()

        x_train, x_test, y_train, y_test = self._get_test_train_splits(
            features_known, classes_known)

        k = self._get_k(classes_known)

        scorer = metrics.make_scorer(self.score_method)

        search = self.hp_method(
            self.clf,
            self.hyperparams,
            n_jobs=-1,
            scoring=scorer,
            cv=StratifiedKFold(n_splits=k))

        search.fit(x_train, y_train)

        self.best_model = search.best_estimator_

        self._set_bests(
            x_train, x_test, y_train, y_test, classes_known, )

        logging.info("  found best hyperparameters (as follows):")
        for param, val in search.best_params_.items():
            logging.info("    {}: {}".format(param, val))

        if multirun > 1:
            logging.info(
                "  generating {} classifiers using found parameters...".format(
                    multirun)
            )
            for m in range(multirun):

                x_train, x_test, y_train, y_test = \
                    self._get_test_train_splits(features_known, classes_known)

                print(
                    "    testing classifier {} of {}...".format(
                        m + 1, multirun),
                    end="\r")

                model = search.best_estimator_.fit(x_train, y_train)

                mean_score = self._get_model_mean_score(
                    model, x_train, x_test, y_train, y_test)

                if mean_score > self.best_test_score:
                    self.best_model = deepcopy(model)
                    # deep copy ensures best model is saved, otherwise an alias
                    # would be saved and overwritten on next fit\
                    self._set_bests(
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        classes_known,

                    )

        if multirun > 1:
            print("\n")

        self.log_report()

        return self.best_model

        print()


class RandomForestAC(AutoClassifier):
    """
    A prefab Random Forest AutoClassifier, no additional setup required.

    New AutoClassifiers should be able to be defined in the same manner without
    needing to overwrite train(), however it shouldn't be too difficult to
    write a new train() method if required.
    """

    def __init__(self):
        """
        Initialize the Random Forest Auto Classifier.

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
            score_method=metrics.balanced_accuracy_score,
        )
