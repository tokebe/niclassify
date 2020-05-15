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


def train_forest(data_known, metadata_known, class_col="Status", multirun=1):
    """Train a random forest.

    Args:
        data_known (DataFrame): Known Data.
        metadata_known (DataFram): Known Metadata, including class label.
        class_col (str, optional): Name of metadata column containing class
            labels. Defaults to "Status".
        multirun (int, optional): Number of times to run. Defaults to 1.

    Returns:
        RandomForest: The Trained Random Forest.

    """
    logging.info("  obtaining best hyperparameters...")

    x_train, x_test, y_train, y_test = train_test_split(
        data_known, metadata_known[class_col],
        stratify=metadata_known[class_col],
        test_size=0.2)  # previously .15

    best_train_predict = None
    best_test_predict = None
    best_train_ba = 0
    best_test_ba = 0
    best_model = None
    best_train_cm = None
    best_test_cm = None

    # hyperparameters to optimize
    parameters = {
        'max_depth': list(np.linspace(10, 50).astype(int)),
        'min_samples_split': list(np.arange(0.05, 0.5, 0.05))}

    rf = RandomForestClassifier(
        class_weight="balanced", oob_score=True, n_estimators=1000)

    # get number of n splits based on data available
    n = min(metadata_known[class_col].value_counts())
    # set n to 10 if the minimum class label count is >= 10, or to the minimum
    # class label count if it's less
    n = 10 if n >= 10 else n

    # Create the random search
    rs = RandomizedSearchCV(
        rf,
        parameters,
        n_jobs=-1,
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=n))

    rs.fit(x_train, y_train)

    best_model = rs.best_estimator_

    best_train_predict = best_model.predict(x_train)
    best_test_predict = best_model.predict(x_test)

    best_train_ba = metrics.balanced_accuracy_score(
        y_train, best_train_predict)
    best_test_ba = metrics.balanced_accuracy_score(y_test, best_test_predict)

    logging.info("  found best hyperparameters (as follows):")
    for param, val in rs.best_params_.items():
        logging.info("    {}: {}".format(param, val))

    best_train_cm = metrics.confusion_matrix(
        y_train,
        best_train_predict,
        labels=metadata_known[class_col].unique(),
        normalize='all'
    )
    best_test_cm = metrics.confusion_matrix(
        y_test,
        best_test_predict,
        labels=metadata_known[class_col].unique(),
        normalize='all'
    )

    if multirun > 1:
        logging.info(
            "  generating {} classifiers using found parameters...".format(
                multirun))
        for m in range(multirun):

            x_train, x_test, y_train, y_test = train_test_split(
                data_known, metadata_known[class_col],
                stratify=metadata_known[class_col],
                test_size=0.2)  # previously .15

            print(
                "    testing forest {} of {}...".format(m + 1, multirun),
                end="\r")

            model = rs.best_estimator_.fit(x_train, y_train)

            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            train_ba = metrics.balanced_accuracy_score(y_train, train_predict)
            test_ba = metrics.balanced_accuracy_score(y_test, test_predict)

            if test_ba > best_test_ba:
                best_train_predict = train_predict
                best_test_predict = test_predict
                best_train_ba = train_ba
                best_test_ba = test_ba

                best_train_cm = metrics.confusion_matrix(
                    y_train,
                    train_predict,
                    labels=metadata_known[class_col].unique(),
                    normalize='all'
                )
                best_test_cm = metrics.confusion_matrix(
                    y_test,
                    test_predict,
                    labels=metadata_known[class_col].unique(),
                    normalize='all'
                )

                best_model = deepcopy(model)
                # deep copy ensures best model is saved, otherwise an alias
                # would be saved and overwritten on next fit\

    if multirun > 1:
        print("\n")

    logging.info("out-of-bag score: {}".format(best_model.oob_score_))
    logging.info("---")

    logging.info("train set BA: {}".format(best_train_ba))
    labels = metadata_known[class_col].unique()
    # logging.info(best_train_cm)
    for true, pred in np.ndindex(best_train_cm.shape):
        if true == pred:
            continue
        else:
            logging.info(
                "train set: percent of {} mislabeled to {}: {:.2f}%".format(
                    labels[true],
                    labels[pred],
                    (best_train_cm[true, pred] * 100)))
    logging.info("---")
    logging.info("test set BA : {}".format(best_test_ba))
    # logging.info(best_test_cm)
    for true, pred in np.ndindex(best_test_cm.shape):
        if true == pred:
            continue
        else:
            logging.info(
                "test set: percent of {} mislabeled to {}: {:.2f}%".format(
                    labels[true],
                    labels[pred],
                    (best_test_cm[true, pred] * 100)))
    logging.info("---")

    return best_model
