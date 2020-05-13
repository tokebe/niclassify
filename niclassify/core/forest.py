try:
    import xlrd

    import numpy as np

    from copy import deepcopy

    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedKFold

    # from itertools import chain, combinations
except ModuleNotFoundError:
    print("Missing required modules. Install requirements by running")
    print("'python -m pip install -r requirements.txt'")


def train_forest(data_known, metadata_known, c_label="Status", multirun=1):
    """Train a random forest.

    Args:
        data_known (DataFrame): Known Data.
        metadata_known (DataFram): Known Metadata, including class label.
        c_label (str, optional): Name of metadata column containing class
            labels. Defaults to "Status".
        multirun (int, optional): Number of times to run. Defaults to 1.

    Returns:
        RandomForest: The Trained Random Forest.

    """
    print("  obtaining best hyperparameters...")

    x_train, x_test, y_train, y_test = train_test_split(
        data_known, metadata_known[c_label],
        stratify=metadata_known[c_label],
        test_size=0.2)  # previously .15

    best_train_predict = None
    best_test_predict = None
    best_train_ba = 0
    best_test_ba = 0
    best_model = None
    train_nerr = 100
    train_ierr = 100
    test_nerr = 100
    test_ierr = 100

    # hyperparameters to optimize
    parameters = {
        # 'n_estimators': np.linspace(500, 1000).astype(int),
        'max_depth': list(np.linspace(10, 50).astype(int)),
        'min_samples_split': list(np.arange(0.05, 0.5, 0.05))}

    rf = RandomForestClassifier(
        class_weight="balanced", oob_score=True, n_estimators=1000)

    # Create the random search
    rs = RandomizedSearchCV(
        rf,
        parameters,
        n_jobs=-1,
        scoring='balanced_accuracy',
        cv=StratifiedKFold(n_splits=10))

    rs.fit(x_train, y_train)

    best_model = rs.best_estimator_

    best_train_predict = best_model.predict(x_train)
    best_test_predict = best_model.predict(x_test)

    best_train_ba = metrics.balanced_accuracy_score(
        y_train, best_train_predict)
    best_test_ba = metrics.balanced_accuracy_score(y_test, best_test_predict)

    if multirun > 1:
        for i in range(multirun):

            x_train, x_test, y_train, y_test = train_test_split(
                data_known, metadata_known[c_label],
                stratify=metadata_known[c_label],
                test_size=0.2)  # previously .15

            print(
                "    testing forest {} of {}...".format(i + 1, multirun),
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

                train_nerr = (y_train[
                    (y_train == "native")
                    & (train_predict == "introduced")].shape[0]
                    / y_train[
                    y_train == "native"].shape[0]) * 100
                train_ierr = (y_train[
                    (y_train == "introduced")
                    & (train_predict == "native")].shape[0]
                    / y_train[
                        y_train == "introduced"].shape[0]) * 100

                test_nerr = (y_test[
                    (y_test == "native")
                    & (test_predict == "introduced")].shape[0]
                    / y_test[
                        y_test == "native"].shape[0]) * 100
                test_ierr = (y_test[
                    (y_test == "introduced")
                    & (test_predict == "native")].shape[0]
                    / y_test[
                        y_test == "introduced"].shape[0]) * 100

                best_model = deepcopy(model)
                # deep copy ensures best model is saved, otherwise an alias
                # would be saved and overwritten on next fit\

    print("  found best hyperparameters (as follows):")
    for key, val in rs.best_params_.items():
        print("    {}: {}".format(key, val))

    print("out-of-bag score: {}".format(best_model.oob_score_))
    print("---")

    # TODO fix these to calculate off best value and not latest

    print("train set BA: {}".format(best_train_ba))
    print(
        "train set: Percent of Native mislabled to Introduced: {:.2f}%".format(
            train_nerr))
    print(
        "train set: Percent of Introduced mislabled to Native: {:.2f}%".format(
            train_ierr))
    print("---")
    print("test set BA : {}".format(best_test_ba))
    print(
        "test set: Percent of Native mislabled to Introduced: {:.2f}%".format(
            test_nerr))
    print(
        "test set: Percent of Introduced mislabled to Native: {:.2f}%".format(
            test_ierr))

    return best_model
