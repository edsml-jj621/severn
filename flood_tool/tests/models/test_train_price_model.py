"""Test Module."""

import flood_tool
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from flood_tool.models import train_price_model


DEFAULT_FILE = (os.path.dirname(__file__)
                + '/../../resources/postcodes_sampled.csv')


def test_get_model():
    """
    Test the function that trains and returns a regressor Model
    based on the method identifier.
    """
    data = pd.read_csv(DEFAULT_FILE)
    inputs = ["lr", "dt"]
    outputs = [LinearRegression,
               DecisionTreeRegressor]
    for i, o in zip(inputs, outputs):
        model = train_price_model.get_model(i, data)
        assert isinstance(model, o)


if __name__ == "__main__":
    test_train_model()
