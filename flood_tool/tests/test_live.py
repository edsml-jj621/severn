"""Test live.py"""

from flood_tool import live
import numpy as np
import math

from pytest import mark


def test_get_closest_station_ref_by_type_from_lat_lng():

    ref = live.get_closest_station_ref_by_type_from_lat_lng(52.872902, -1.496148)
    assert ref == '3100'

if __name__ == "__main__":
    test_get_closest_station_ref_by_type_from_lat_lng()
