"""Test analysis.py"""

from flood_tool import analysis
import numpy as np
import math

from pytest import mark


def test_plot_property_value_histogram():

    data = analysis.plot_property_value_histogram()

    assert np.isclose(data.size, 1000).all()


if __name__ == "__main__":
    test_plot_property_value_histogram()
