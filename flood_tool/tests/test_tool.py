"""Test Module."""

import flood_tool
import numpy as np
import math

from pytest import mark


tool = flood_tool.Tool()


def test_get_easting_northing():
    """
    Test the function that gets a frame of OS eastings
    and northings from a collection of input postcodes.
    """

    data = tool.get_easting_northing(['YO62 4LS', 'DE2 3DA'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].easting, 467631.0).all()
    assert np.isclose(data.iloc[0].northing, 472825.0).all()
    assert np.isclose(data.iloc[1].easting, 434011.0).all()
    assert np.isclose(data.iloc[1].northing, 330722.0).all()


def test_get_lat_long():
    """
    Test the function that gets a frame containing GPS latitude
    and longitude information for a collection of of postcodes.
    """

    data = tool.get_lat_long(['YO62 4LS', 'SOE 35', 'NR32 2NF', 'XX1 2XX'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].latitude, 54.147, 1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -0.966, 1.0e-3).all()
    assert math.isnan(data.iloc[1].latitude)


def test_get_postcode_from_sector():
    '''
    Test the function that gets a frame of
    postcodes from a collection of input sectors.
    '''
    data = tool.get_postcode_from_sector(['TN13 2', 'GU1 4', 'CH60 0W'])

    assert math.isnan(data.iloc[0].postcode)
    assert math.isnan(data.iloc[2].postcode)
    assert data.iloc[1].postcode == "GU1 4GR"


def test_get_flood_class():
    '''
    Test the function that generates series predicting flood
    probability classification for a collection of poscodes.
    '''
    postcodes = tool.postcode_df["postcode"]
    riskLabel_df = tool.get_flood_class(postcodes, "knn")
    assert len(riskLabel_df.unique()) > 3 and len(riskLabel_df.unique()) <= 10
    riskLabel_df = tool.get_flood_class(postcodes, "dt")
    assert len(riskLabel_df.unique()) > 3 and len(riskLabel_df.unique()) <= 10


def test_get_median_house_price_estimate():
    '''
    Test the function that generates series predicting
    median house price for a collection of poscodes.
    '''
    postcodes = tool.postcode_df["postcode"]
    median_house_price_df = tool.get_median_house_price_estimate(
        postcodes, "lr")
    assert len(median_house_price_df) == 1000
    assert np.isclose(median_house_price_df.median(), 389576).all()


def test_get_total_value():
    '''
    Test the function that returns a series of estimates of the total property values
    of a collection of postcode units or sectors.
    '''
    postcodes = tool.postcode_df['postcode']
    total_value_postcode_df = tool.get_total_value(postcodes)
    assert len(total_value_postcode_df) == 1000
    assert np.isclose(total_value_postcode_df.mean(), 386948).all()


def test_get_annual_flood_risk():
    '''
    Test the function that returns a series of estimates of the risk for
    each postcode
    '''
    postcodes = tool.postcode_df['postcode']
    risk_df = tool.get_annual_flood_risk(postcodes)
    assert len(risk_df) == 1000
    assert np.isclose(risk_df.median(), 2.0309).all()


if __name__ == "__main__":
    test_get_easting_northing()
