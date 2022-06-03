"""Preprocess postcodes sampled data, then train and save multiple classifier models
i.e. KNN, RandomForest, DecisionTree
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_model(file, output_folder):
    """Train model for predicting flood risk class.

    Parameters
    ----------
    file : str
        labelled postcodes file with columns `easting`, `northing`, `riskLabel`

    output_folder : str, optional
        folder to store output models if specify
    """
    data = pd.read_csv(file)
    data = preprocess_data(data, save_file=True, output_folder="output")

    # KNN, Random Forest, Decision Tree, GradientBoostingClassifier, AdaBoost
    filenames = {
        "knn": "knn_model.sav",
        "rdmf": "random_forest_model.sav",
        "dt": "decision_tree_model.sav",
        "gdbt": "gradient_boosting_model.sav",
        "ada": "ada_model.sav"
    }
    for name in ["knn", "rdmf", "dt", "ada"]:
        model = get_model(name, data, evaluate=True)
        # save the model to disk
        if name == "rdmf":
            joblib.dump(model, f"{output_folder + '/' if output_folder else output_folder}{filenames[name]}",
                        compress=3)  # compress large models
        else:
            joblib.dump(
                model, f"{output_folder + '/' if output_folder else output_folder}{filenames[name]}")


def get_model(name, data, evaluate=False):
    """Train and return model given name (identifier)

    Parameters
    ----------
    name : str
        model identifier

    data : pandas.Dataframe
        labelled postcodes data with `riskLabel`

    evaluate : bool, optional (default=False)
        if True, evaluate the model

    Returns
    -------
    class
        trained model
    """
    X = data[["easting", "northing"]]
    y = data["riskLabel"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3)
    model = {
        "knn": KNeighborsClassifier(n_neighbors=1),
        "rdmf": RandomForestClassifier(),
        "dt": DecisionTreeClassifier(),
        "gdbt": GradientBoostingClassifier(max_depth=2, n_estimators=150),
        "ada": AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=8), n_estimators=500)
    }[name]
    model.fit(X_train, y_train)

    if evaluate:
        y_pred = model.predict(X_test)
        print("=====Evaluation for: " + name + "========")
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None)
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("accuracy: " + str(accuracy))
        print("f1: " + str(f1))
    return model


def preprocess_data(data, save_file=False, output_folder=""):
    """Preprocess model data
    - remove outliers

    Parameters
    ----------
    data : pandas.DataFrame
        data with `riskLabel`

    save_file : bool, optional (default=False)
        if true, save processed data file

    output_folder : str, optional
        folder to store output processed data file if specify

    Returns
    -------
    pandas.DataFrame
    """
    group_count = data.groupby("riskLabel")[
        "riskLabel"].count().sort_values(ascending=False)
    major_cls, major_count = (
        group_count.index[0], group_count.values[0])  # majority class and count

    # Random Over Sampler
    over_ratio = 0.9  # ratio of minority to majority after oversampling
    oversampling_size = int(over_ratio * major_count)
    data = random_over_sampler(
        data, oversampling_size, major_cls)
    group_count = data.groupby("riskLabel")[
        "riskLabel"].count().sort_values(ascending=False)

    # Random Under Sampler
    under_ratio = 1  # ratio of minority to majority after undersampling
    undersampling_size = int(oversampling_size / under_ratio)
    data = random_under_sampler(
        data, undersampling_size, major_cls)

    if save_file:
        data.to_csv(
            f"{output_folder + '/' if output_folder else ''}postcodes_sampled_processed.csv")
    return data


def random_over_sampler(data, size, major_cls):
    """Random over sampling
    Duplicate random rows of minority classes

    Parameters
    ----------
    data : pandas.DataFrame

    size : float
        size of minority classes after oversampling

    major_cls : int
        the majority class value

    Returns
    -------
    pandas.DataFrame
    """
    lst = [data]
    for class_index, gp in data.groupby('riskLabel'):
        if class_index != major_cls:
            lst.append(gp.sample(size-len(gp), replace=True))
            data_oversampled = pd.concat(lst)
    return data_oversampled


def random_under_sampler(data, size, major_cls):
    """Random under sampling

    Parameters
    ----------
    data : pandas.DataFrame

    size : float
        size of majority classes after undersampling

    major_cls : int
        the majority class value

    Returns
    -------
    pandas.DataFrame
    """
    major_cls_idx = data[data.riskLabel == major_cls].index
    rdm_major_cls_idx = np.random.choice(
        major_cls_idx, len(major_cls_idx) - size, replace=False)  # select random indices
    data_undersampled = data.drop(rdm_major_cls_idx)  # drop rows
    return data_undersampled
