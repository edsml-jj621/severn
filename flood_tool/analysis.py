"""Analysis tools."""

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from . import live
from .tool import Tool

__all__ = [
    'plot_postcode_density',
    'plot_flood_risk_histogram',
    'plot_flood_class_histogram',
    'plot_property_value_histogram'
]

DEFAULT_FILE = (os.path.dirname(__file__)
                + '/resources/postcodes_unlabelled.csv')


def plot_postcode_density(postcode_file=DEFAULT_FILE,
                          coordinate=['easting', 'northing'], dx=1000):

    pdb = pd.read_csv(postcode_file)

    bbox = (pdb[coordinate[0]].min()-0.5*dx, pdb[coordinate[0]].max()+0.5*dx,
            pdb[coordinate[1]].min()-0.5*dx, pdb[coordinate[1]].max()+0.5*dx)

    nx = (math.ceil((bbox[1]-bbox[0])/dx),
          math.ceil((bbox[3]-bbox[2])/dx))

    x = np.linspace(bbox[0]+0.5*dx, bbox[0]+(nx[0]-0.5)*dx, nx[0])
    y = np.linspace(bbox[2]+0.5*dx, bbox[2]+(nx[1]-0.5)*dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x-bbox[0])/dx), math.floor((y-bbox[2])/dx)] += 1

    plt.pcolormesh(X, Y, np.where(Z > 0, Z, np.nan).T,
                   norm=matplotlib.colors.LogNorm())
    plt.axis('equal')
    plt.colorbar()


def plot_flood_risk_histogram(postcode_file=DEFAULT_FILE):
    """
    Plot predicted annual flood risk histogram from postcode .csv file.

    Parameters
    ----------
    postcode_file: a .csv file contains postcode column

    Returns
    -------
    histogram figure

    Examples
    --------
    >>> analysis.plot_flood_risk_histogram()
    """
    t = Tool()
    df = pd.read_csv(postcode_file)
    postcodes = df['postcode'].values

    # Get flood risk
    Flood_risk = t.get_annual_flood_risk(postcodes)

    # plot histogram
    plt.figure(figsize=(8, 6))
    plt.title("annual flood risk distribution")
    plt.xlabel("annual flood risk")
    plt.ylabel("counts")
    plt.hist(Flood_risk, rwidth=0.8)
    return Flood_risk


def plot_flood_class_histogram(postcode_file=DEFAULT_FILE, method=0):
    """
    Plot predicted flood class histogram from postcode .csv file.

    Parameters
    ----------
    postcode_file: a .csv file contains postcode column
    method: "dt", "knn", "rdmf", "ada"
        Decision Tree, KNN, Random Forest, AdaBoost

    Returns
    -------
    histogram figure

    Examples
    --------
    >>> analysis.plot_flood_class_histogram()
    """
    t = Tool()
    df = pd.read_csv(postcode_file)
    postcodes = df['postcode'].values

    # Get flood class
    flood = t.get_flood_class(postcodes, method)

    # plot histogram
    plt.figure(figsize=(8, 6))
    plt.title("flood class distribution")
    plt.xlabel("flood class")
    plt.ylabel("counts")
    plt.hist(flood, bins=10, rwidth=0.8)
    return flood


def plot_property_value_histogram(postcode_file=DEFAULT_FILE, method=0):
    """
    Plot predicted median property value histogram from postcode .csv file.

    Parameters
    ----------
    postcode_file: a .csv file contains postcode column
    method: "lr", "dt", "rfr", "sv"
        Linear Regression, Decision Tree, Random Forest, SV

    Returns
    -------
    histogram figure

    Examples
    --------
    >>> analysis.plot_property_value_histogram()
    """
    t = Tool()
    df = pd.read_csv(postcode_file)
    postcodes = df['postcode'].values

    # Get predicted median house price
    value = t.get_median_house_price_estimate(postcodes, method)

    # plot histogram
    plt.figure(figsize=(8, 6))
    plt.title("median house price distribution")
    plt.xlabel("property value")
    plt.ylabel("counts")
    plt.hist(value, bins=10, rwidth=0.8)
    return value


def plot_reading(file, parameter, station_ref):
    day_level_df = live.get_station_reading(file, parameter, station_ref)
    day_level_df["dateTime"] = pd.to_datetime(day_level_df["dateTime"])
    day_level_df["dateTime"] = day_level_df["dateTime"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    interpolated = day_level_df.interpolate(method='linear')
    title_param = "Rainfall" if parameter == "rainfall" else "Water level"
    unit = "mm" if parameter == "rainfall" else "mASD"
    ax = interpolated.plot(x="dateTime")
    # ax.xaxis_date()
    # ax.figure.autofmt_xdate(rotation=0, ha='center')
    ax.set_title(f"{title_param} of a day at stations")
    ax.set_xlabel("dataTime")
    ax.set_ylabel(unit)


def get_combined_data(postcode_file, reading_file):
    """Combine reading files and postcode files to get rainfall, water level,
    flood probability... indexed by postcodes.
    """
    tool = Tool(postcode_file)
    postcode_df = tool.get_combined_data()

    # rainfall / water level data indexed by station reference
    rainfall_df = live.get_grouped_reading(reading_file, "rainfall")
    level_df = live.get_grouped_reading(reading_file, "level")

    # Get station reference of postcodes from lat, long
    def get_station_ref(s, s_type):
        stationRef = live.get_closest_station_ref_by_type_from_lat_lng(
            s["latitude"], s["longitude"], s_type)
        return stationRef
    postcode_df["rainfallStationRef"] = postcode_df.apply(
        get_station_ref, args=("rainfall",), axis=1)
    postcode_df["levelStationRef"] = postcode_df.apply(
        get_station_ref, args=("level",), axis=1)

    # Join postcode data, rainfall data and level data together by station reference
    postcode_df = postcode_df.join(rainfall_df, on='rainfallStationRef').rename(
        columns={'value': 'sumRainFall'})
    postcode_df = postcode_df.join(level_df, on='levelStationRef').rename(
        columns={'value': 'maxRiverLevel'})
    return postcode_df
