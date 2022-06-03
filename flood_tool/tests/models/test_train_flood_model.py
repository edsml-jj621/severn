"""Test Module."""

import flood_tool
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from flood_tool.models import train_flood_model

DEFAULT_FILE = (os.path.dirname(__file__)
                + '/../../resources/postcodes_sampled.csv')


def test_get_model():
    """
    Test the function that trains and returns a classifier Model
    based on the method identifier.
    """
    data = pd.read_csv(DEFAULT_FILE)
    inputs = ["knn", "dt"]
    outputs = [KNeighborsClassifier,
               DecisionTreeClassifier]
    for i, o in zip(inputs, outputs):
        model = train_flood_model.get_model(i, data)
        assert isinstance(model, o)


def test_preprocess_data():
    """
    Test the function that preprocess unlabelled data file.
    """
    data = pd.read_csv(DEFAULT_FILE)
    preprocessed_data = train_flood_model.preprocess_data(data)

    assert len(preprocessed_data) > len(data)
    # check upsampling
    assert len(preprocessed_data[preprocessed_data["riskLabel"] ==
                                 10]) > len(data[data["riskLabel"] == 10])
    # check downsampling
    assert len(preprocessed_data[preprocessed_data["riskLabel"] ==
                                 1]) < len(data[data["riskLabel"] == 1])


def test_random_over_sampler():
    """
    Test the function that runs upsampling on data.
    """
    data = pd.read_csv(DEFAULT_FILE)
    # get riskLabel majority class and size
    group_count = data.groupby("riskLabel")[
        "riskLabel"].count().sort_values(ascending=False)
    size = group_count.values[0]
    majority_class = group_count.index[0]
    data = train_flood_model.random_over_sampler(data, size, majority_class)
    assert len(data[data["riskLabel"] == 10]) == size


def test_random_under_sampler():
    """
    Test the function that runs downsampling on data.
    """
    data = pd.read_csv(DEFAULT_FILE)
    # get riskLabel majority class and size
    group_count = data.groupby("riskLabel")[
        "riskLabel"].count().sort_values(ascending=False)
    size = group_count.values[0] - 1000
    majority_class = group_count.index[0]
    data = train_flood_model.random_under_sampler(data, size, majority_class)
    assert len(data[data["riskLabel"] == 1]) == size


if __name__ == "__main__":
    test_train_model()
