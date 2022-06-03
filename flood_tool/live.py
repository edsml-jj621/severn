"""Interactions with rainfall and river data."""

import numpy as np
import pandas as pd
import os
from numpy import sin, arcsin, cos, sqrt, deg2rad

__all__ = ["get_station_data",
           "get_closest_station_ref_from_lat_lng",
           "get_closest_station_ref_by_type_from_lat_lng",
           "get_rainfall_riverlevel_from_lat_lng",
           "get_rainfall_classifier_from_lat_lng",
           "get_station_reading",
           "get_grouped_reading"]

STATION_FILE = (os.path.dirname(__file__)
                + '/resources/stations.csv')
STATION_DF = pd.read_csv(STATION_FILE)


def get_station_data(filename, station_reference):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_station_data('flood_tool/resources/wet_day.csv)
    """
    frame = pd.read_csv(filename)
    frame = frame.loc[frame.stationReference == station_reference]

    return pd.to_numeric(frame.value.values)


def get_closest_station_ref_from_lat_lng(lat, lng):
    """Return stationReference where the station is
    closest to the given lat, lng.

    Parameters
    ----------
    lat: float
        latitude
    lng: float
        longitude

    Returns
    -------
    ref: str
    stationReference

    Example
    -------
    >>> ref = get_closest_station_ref_from_lat_lng(52.872902, -1.496148)
    """
    frame = STATION_DF
    lng1 = frame['longitude']
    lat1 = frame['latitude']
    lat2 = lat
    lng2 = lng
    # deg to rad
    lng1 = deg2rad(lng1)
    lat1 = deg2rad(lat1)
    lng2 = deg2rad(lng2)
    lat2 = deg2rad(lat2)
    # difference
    dlon = lng2-lng1
    dlat = lat2-lat1
    # haversine formula to calculate the distance
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*arcsin(sqrt(a))*6371.393*1000  # earth's avg radius，6371.393km
    # get the min distance index and its ref
    ref = frame.iloc[np.argmin(distance)]['stationReference']

    return ref


def get_closest_station_ref_by_type_from_lat_lng(lat, lng, s_type="rainfall"):
    """Return stationReference (rainfall or level) where the station is
    closest to the given lat, lng.

    Parameters
    ----------
    lat: float
        latitude
    lng: float
        longitude
    type: str
        rainfall / level station

    Returns
    -------
    ref: str
    stationReference

    Example
    -------
    >>> ref = get_closest_station_ref_from_lat_lng(52.872902, -1.496148)
    """
    frame = STATION_DF
    if s_type == "rainfall":
        frame = frame[frame["stationName"] == "Rainfall station"]
    else:
        frame = frame[~(frame["stationName"] == "Rainfall station")]
    lng1 = frame['longitude']
    lat1 = frame['latitude']
    lat2 = lat
    lng2 = lng
    # deg to rad
    lng1 = deg2rad(lng1)
    lat1 = deg2rad(lat1)
    lng2 = deg2rad(lng2)
    lat2 = deg2rad(lat2)
    # difference
    dlon = lng2-lng1
    dlat = lat2-lat1
    # haversine formula to calculate the distance
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*arcsin(sqrt(a))*6371.393*1000  # earth's avg radius，6371.393km
    # get the min distance index and its ref
    ref = frame.iloc[np.argmin(distance)]['stationReference']
    return ref


def get_rainfall_riverlevel_from_lat_lng(filename, lat, lng):
    """Return total rainfall and max river level where given the
    latitude and longitude in a day.

    Parameters
    ----------
    filename: str
        typical_day or wet_day
    lat: float
        latitude
    lng: float
        longitude

    Returns
    -------
    total_rainfall: float
        total rainfall of the day (unit: mm)
    max_level: float
        max of the river level of the day (unit: mASD)

    Example
    -------
    >>> rainfall, level = get_rainfall_riverlevel_from_lat_lng('flood_tool/resources/wet_day.csv', 52.872902, -1.496148)
    """
    ref = get_closest_station_ref_from_lat_lng(lat, lng)

    df_day = pd.read_csv(filename, low_memory=False)
    max_level = df_day.loc[(df_day['stationReference'] == ref) & (
        df_day['parameter'] == 'level')]['value'].max()

    rainfall = df_day.loc[(df_day['stationReference'] == ref) & (
        df_day['parameter'] == 'rainfall')]['value'].values
    total_rainfall = sum(pd.to_numeric(rainfall))

    return total_rainfall, max_level


def get_rainfall_classifier_from_lat_lng(filename, lat, lng):
    """Return rainfall class of the station
    that is the closest to the given lat, lng.

    Parameters
    ----------
    lat: float
        latitude
    lng: float
        longitude

    Returns
    -------
    rain_class: str
    rainfall(mm) class: no rain, slight, moderate, heavy, violent
    """
    # find the nearest station reference
    ref = get_closest_station_ref_from_lat_lng(lat, lng)

    df = pd.read_csv(filename, low_memory=False)

    value = df.loc[(df['stationReference'] == ref) & (
        df['parameter'] == 'rainfall')]['value'].values
    value = pd.to_numeric(value)

    if value.size == 0:
        return 'no data'

    step = 4

    num = len(value)
    num = num - num % 4
    rainfall = [value[i:i+step] for i in range(0, num, step)]

    # count total amount of rainfall in an hour
    rain_per_hour = np.sum(rainfall, axis=1)

    rain_per_hour_max = max(rain_per_hour)

    # classify the max rainfall in a day
    if rain_per_hour_max == 0:
        rain_class = "no rain"
    elif rain_per_hour_max < 2:
        rain_class = "slight"
    elif rain_per_hour_max < 4:
        rain_class = "moderate"
    elif rain_per_hour_max < 50:
        rain_class = "heavy"
    else:
        rain_class = "violent"

    return rain_class


def get_rainfall_per_hour(filename, lat, lng):
    """Return rainfall class of the station
    that is the closest to the given lat, lng.

    Parameters
    ----------
    lat: float
        latitude
    lng: float
        longitude

    Returns
    -------
    rain_per_hour: list
    """
    # find the nearest station reference
    ref = get_closest_station_ref_from_lat_lng(lat, lng)

    df = pd.read_csv(filename, low_memory=False)

    value = df.loc[(df['stationReference'] == ref) & (
        df['parameter'] == 'rainfall')]['value'].values
    value = pd.to_numeric(value)

    if value.size == 0:
        return 'no data'

    step = 4
    num = len(value)
    num = num - num % 4
    rainfall = [value[i:i+step] for i in range(0, num, step)]

    # count total amount of rainfall in an hour
    rain_per_hour = np.sum(rainfall, axis=1)

    return rain_per_hour


def get_station_reading(file, parameter, station_ref):
    """Get rainfall or water level data for a list of staions.
    Return DataFrame indexed by dateTime, with station name as the column names.

    Parameters
    ----------
    file : str
        filename / path of the rainfall & water level reading file
    parameter : str
        rainfall or level
    station_ref : list
        list of station_ref

    Example
    -------
    >>> get_station_reading(['4761', 'E2694', 'E4823'])
                     dateTime   4761  E2694  E4823
    0    2021-05-07T00:00:00Z  0.069  0.056    NaN
    1    2021-05-07T00:00:01Z    NaN    NaN  0.325
    """
    day_df = pd.read_csv(file, low_memory=False)
    mask = (day_df["parameter"] == parameter) & (
        day_df["stationReference"].isin(station_ref))
    day_reading_df = day_df[mask][["dateTime", "stationReference", "value"]]
    if parameter == "rainfall":
        day_reading_df["value"] = day_reading_df["value"].map(
            lambda x: str(x).split("|")[0])  # fix str |
    day_reading_df.value = day_reading_df.value.astype(float)
    day_reading_df = day_reading_df.drop_duplicates(
        subset=['dateTime', 'stationReference'], keep=False)
    day_reading_df = day_reading_df.set_index(["dateTime", "stationReference"]).unstack()[
        "value"].rename_axis(None, axis=1).reset_index()
    return day_reading_df


def get_grouped_reading(file, parameter):
    """Get sum of rainfall / max water level indexed by station reference

    Example
    -------
    >>> get_grouped_reading("wet_day.csv", "rainfall")

                        value
    stationReference
    000008              0.0
    000028              0.2
    000075TP            0.0
    """
    day_df = pd.read_csv(file, low_memory=False)
    day_df["value"] = day_df["value"].map(
        lambda x: str(x).split("|")[0])  # fix str |
    day_df.value = day_df.value.astype(float)
    if parameter == "rainfall":
        grouped_df = day_df.groupby(
            ["stationReference", "parameter"]).value.sum()
    else:
        grouped_df = day_df.groupby(
            ["stationReference", "parameter"]).value.max()
    grouped_df = grouped_df[grouped_df.index.get_level_values(
        'parameter') == parameter]
    grouped_df = grouped_df.droplevel("parameter").to_frame()
    return grouped_df
