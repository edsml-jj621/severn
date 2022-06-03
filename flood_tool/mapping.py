import folium
import pandas as pd
import numpy as np
from . import tool
from . import live
from . import geo
import os
from folium.plugins import HeatMap
from math import isnan

__all__ = [
    'plot_circle', 'plot_popping', 'plot_heatmap',
    'plot_flood_prob_level_with_given_postcode', 'plot_house_price',
    'plot_house_price_sampled', 'plot_flood_prob_sampled', 'plot_24h_rainfall'
]

t = tool.Tool()

READING_FILE = (os.path.dirname(__file__)
                + '/resources/wet_day.csv')
POSTCODES_SAMPLED_FILE = (os.path.dirname(__file__)
                          + '/resources/postcodes_sampled.csv')


def plot_circle(lat, lon, radius, map=None, **kwargs):
    """
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> armageddon.plot_circle(52.79, -2.95, 1e3, map=None)
    """

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle(location=[lat, lon],
                  radius=radius,
                  fill=True,
                  fillOpacity=0.6,
                  **kwargs).add_to(map)

    return map


def plot_popping(postcodes):
    """
    Plot popping up window displaying all information required.

    Parameters
    ----------
    postcodes: sequence of string
        Sequence of postcodes.

    Returns
    -------
    Folium map object
        Click on markers will show detailed information on those
    points, including rainfall, rainfall class, river Level,
    flood event probability, property value and flood risk

    Examples
    --------
    >>> mapping.plot_popping(['DE2 3DA', 'LN5 7RW'])
    """
    table = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table {{
        width:100%;
    }}
    table, th, td {{
        border: 1px solid black;
        border-collapse: collapse;
    }}
    th, td {{
        padding: 5px;
        text-align: left;
    }}
    table#t01 tr:nth-child(odd) {{
        background-color: #eee;
    }}
    table#t01 tr:nth-child(even) {{
       background-color:#fff;
    }}
    </style>
    </head>
    <body>

    <table id="t01">
      <tr>
        <td>Rainfall</td>
        <td>{}</td>
      </tr>
      <tr>
        <td>Rainfall Class</td>
        <td>{}</td>
      </tr>
      <tr>
        <td>River Level</td>
        <td>{}</td>
      </tr>
      <tr>
        <td>Flood Event Probability</td>
        <td>{}</td>
      </tr>
      <tr>
        <td>Property Value</td>
        <td>{}</td>
      </tr>
      <tr>
        <td>Flood Risk</td>
        <td>{}</td>
      </tr>
    </table>
    </body>
    </html>
    """.format
    # Get lat lng
    t = tool.Tool()
    # df = pd.read_csv('resources/postcodes_unlabelled.csv')
    t.get_lat_long(postcodes)

    # Property price
    pred_price = t.get_median_house_price_estimate(postcodes)

    # Flood event
    pred_flood_class = t.get_flood_class(postcodes)
    Label = pd.Series(data=pred_flood_class,
                      index=np.asarray(postcodes),
                      name='riskLabel')
    Flood_event = Label.replace({
        1: 0.01,
        2: 0.05,
        3: 0.1,
        4: 0.5,
        5: 1,
        6: 1.5,
        7: 2,
        8: 3,
        9: 4,
        10: 5
    })

    # Get flood risk
    Flood_risk = t.get_annual_flood_risk(postcodes)

    # coord = mapping.get_lat_long_from_postcode(postcodes)
    # Get the lat and lng
    coordinates = t.get_lat_long(postcodes)

    # Get the rainfall and rainfall_class and river_level
    total_rain = []
    total_rainclass = []
    total_riverlevel = []

    for i in range(len(postcodes)):
        rainfall, maxlevel = live.get_rainfall_riverlevel_from_lat_lng(
            READING_FILE, coordinates['latitude'][i],
            coordinates['longitude'][i])
        total_rain.append(rainfall)
        total_riverlevel.append(maxlevel)

        rain_class = live.get_rainfall_classifier_from_lat_lng(
            READING_FILE, coordinates['latitude'][i],
            coordinates['longitude'][i])
        total_rainclass.append(rain_class)

    from folium import IFrame

    map = folium.Map(location=[54, -1], zoom_start=6, tiles='OpenStreetMap')

    for i in range(len(postcodes)):

        iframe = IFrame(html=table(
            str(total_rain[i]) + " mm", total_rainclass[i],
            str(total_riverlevel[i]) + " mASD",
            str(Flood_event[i]) + "%", str(pred_price[i]), str(Flood_risk[i])),
            width=420,
            height=280)

        popup = folium.Popup(iframe, max_width=420)

        folium.Marker(location=(coordinates['latitude'][i],
                                coordinates['longitude'][i]),
                      popup=popup,
                      icon=folium.Icon(color='black',
                                       icon='info-sign')).add_to(map)

    return map


def plot_heatmap(postcodes):
    """
    Plot heatmap of max river level and total rainfall in a day
    from a given sequence of postcodes.

    Parameters
    ----------
    postcodes: sequence of string
        Sequence of postcodes.

    Returns
    -------
    Folium heatmap object

    Examples
    --------
    >>> mapping.plot_heatmap(['DE2 3DA', 'LN5 7RW'])
    """
    t = tool.Tool()
    lat_long = t.get_lat_long(postcodes)
    total_rain = []
    total_riverlevel = []
    for i in range(len(lat_long)):
        rainfall, maxlevel = live.get_rainfall_riverlevel_from_lat_lng(
            READING_FILE, lat_long.latitude[i], lat_long.longitude[i])
        total_rain.append(rainfall)
        total_riverlevel.append(maxlevel)

    rain_map = folium.Map(location=[54, -1],
                          zoom_start=6,
                          tiles='OpenStreetMap')

    dicts = {0.1: '#CCFFFF', 0.65: 'lime', 1: 'blue'}
    heatdata_rain = [[
        lat_long.latitude[i], lat_long.longitude[i], total_rain[i]
    ] for i in range(len(lat_long))]
    HeatMap(heatdata_rain, gradient=dicts).add_to(rain_map)

    river_map = folium.Map(location=[54, -1],
                           zoom_start=6,
                           tiles='OpenStreetMap')

    # dicts1 = {0.1: 'red', 0.65: 'lime', 1: 'blue'}
    heatdata_river = [[
        lat_long.latitude[i], lat_long.longitude[i], total_riverlevel[i]
    ] for i in range(len(lat_long))]
    heatdata_river = [[0 if isnan(float(k)) else float(k) for k in i]
                      for i in heatdata_river]
    HeatMap(heatdata_river).add_to(river_map)

    return rain_map, river_map


def plot_flood_prob_level_with_given_postcode(postcodes):
    """
    Plot flood probability map from given list of postcodes.

    Parameters
    ----------
    postcode:sequence of strs
        Sequence of postcodes.

    Returns
    -------
    Folium map object
        Different colours of circle markers show
        different flood probabilities.

    Examples
    --------
    >>> mapping.plot_flood_prob_level_with_given_postcode(['HP27 0BF', 'W5 2BX'])
    """
    t = tool.Tool()
    pred_flood_class = t.get_flood_class(postcodes)
    lat_long = t.get_lat_long(postcodes)
    pred_flood_class
    # Flood_event = pred_flood_class.replace({
    #     1: 0.01,
    #     2: 0.05,
    #     3: 0.1,
    #     4: 0.5,
    #     5: 1,
    #     6: 1.5,
    #     7: 2,
    #     8: 3,
    #     9: 4,
    #     10: 5
    # })

    map = folium.Map(location=[54, -1], zoom_start=7, tiles='OpenStreetMap')

    for num in range(len(lat_long)):
        if pred_flood_class[num] == 1:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup={
                    pred_flood_class.index[num]: 'Risk:0.01%'
                },
                color='#FF9900',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 2:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:0.05%',
                color='#FF6600',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 3:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:0.1%',
                color='#FF3300',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 4:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:0.5%',
                color='#FF0000',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 5:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:1%',
                color='#CC3300',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 6:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:1.5%',
                color='#CC0000',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 7:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:2%',
                color='#993300',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 8:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:3%',
                color='#990000',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 9:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:4%',
                color='#990033',
                fill=True).add_to(map)

        elif pred_flood_class[num] == 10:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=1500,
                popup='Risk:5%',
                color='#990033',
                fill=True).add_to(map)
    return map


def plot_house_price(postcodes):
    """
    Plot house price map from given list of postcodes.

    Parameters
    ----------
    postcode:sequence of strs
        Sequence of postcodes.

    Returns
    -------
    Folium map object
        Different colours of circle markers show
    different house price ranges.

    Examples
    --------
    >>> mapping.plot_house_price(['HP27 0BF', 'W5 2BX'])
    """
    t = tool.Tool()
    pred_price = t.get_median_house_price_estimate(postcodes)
    lat_long = t.get_lat_long(postcodes)

    houseprice_map = folium.Map(location=[54, -1],
                                zoom_start=6,
                                tiles='OpenStreetMap')

    for num in range(len(lat_long)):
        if pred_price[num] < 250000:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=5,
                popup={
                    pred_price.index[num]: '<25000'
                },
                color='lightblue',
                fill=True).add_to(houseprice_map)

        elif pred_price[num] < 375000:
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=5,
                popup={
                    pred_price.index[num]: '<25000'
                },
                color='cadetblue',
                fill=True).add_to(houseprice_map)

        elif pred_price[num] < 500000:
            # postcode = pred_price.index[num]
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=5,
                popup={
                    pred_price.index[num]: '<25000'
                },
                color='blue',
                fill=True).add_to(houseprice_map)

        else:
            # postcode = pred_price.index[num]
            folium.Circle(
                location=[lat_long.latitude[num], lat_long.longitude[num]],
                radius=5,
                popup={
                    pred_price.index[num]: '<25000'
                },
                color='darkblue',
                fill=True).add_to(houseprice_map)

    return houseprice_map


def plot_house_price_sampled():
    """
    Plot house price map from given sampled csv

    Parameters
    ----------

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> map = plot_house_price_sampled()
    """
    df = pd.read_csv(POSTCODES_SAMPLED_FILE)
    lat, lng = geo.get_gps_lat_long_from_easting_northing(
        df.iloc[:]['easting'], df.iloc[:]['northing'])

    map = folium.Map(location=[54, -1], zoom_start=6, tiles='OpenStreetMap')

    for i in range(df.shape[0]):

        if df.iloc[i]['medianPrice'] < 250000:
            folium.CircleMarker(location=[lat[i], lng[i]],
                                radius=1,
                                fill=True,
                                color='lightblue',
                                fillOpacity=1).add_to(map)

        elif df.iloc[i]['medianPrice'] < 375000:
            folium.CircleMarker(location=[lat[i], lng[i]],
                                radius=1,
                                fill=True,
                                color='cadetblue',
                                fillOpacity=1).add_to(map)

        elif df.iloc[i]['medianPrice'] < 500000:
            folium.CircleMarker(location=[lat[i], lng[i]],
                                radius=1,
                                fill=True,
                                color='blue',
                                fillOpacity=1).add_to(map)

        else:
            folium.CircleMarker(location=[lat[i], lng[i]],
                                radius=1,
                                fill=True,
                                color='darkblue',
                                fillOpacity=1).add_to(map)

    return map


def plot_flood_prob_sampled():
    """
    Plot flood probability map from given sampled csv

    Parameters
    ----------

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> map = plot_flood_prob_sampled()
    """

    map = folium.Map(location=[54, -1], zoom_start=7, tiles='OpenStreetMap')
    df = pd.read_csv(POSTCODES_SAMPLED_FILE)
    lat, lng = geo.get_gps_lat_long_from_easting_northing(
        df.iloc[:]['easting'], df.iloc[:]['northing'])
    for i in range(df.shape[0]):
        if df.iloc[i]['riskLabel'] == 1 & (i % 3 == 0):
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#FF9900').add_to(map)
        elif df.iloc[i]['riskLabel'] == 2:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#FF6600').add_to(map)
        elif df.iloc[i]['riskLabel'] == 3:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#FF3300').add_to(map)
        elif df.iloc[i]['riskLabel'] == 4:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#FF0000').add_to(map)
        elif df.iloc[i]['riskLabel'] == 5:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#CC3300').add_to(map)
        elif df.iloc[i]['riskLabel'] == 6:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#CC0000').add_to(map)
        elif df.iloc[i]['riskLabel'] == 7:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#993300').add_to(map)
        elif df.iloc[i]['riskLabel'] == 8:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#990000').add_to(map)
        elif df.iloc[i]['riskLabel'] == 9:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#990033').add_to(map)
        elif df.iloc[i]['riskLabel'] == 10:
            folium.Circle(location=[lat[i], lng[i]],
                          radius=1,
                          fill=True,
                          color='#990033').add_to(map)
    return map


def plot_24h_rainfall(postcodes):
    """
    Plot rainfall level heatmap for the nearest station of the specific postcodes with a
    24 hour timeline, rainfall level data acquired from wet_day.csv

    Parameters
    ----------
    postcode:sequence of strs
        Sequence of postcodes.

    Returns
    -------

    Animated Folium map object:
    showing daily variation for the rainlevel at nearest station of the specific postcodes


    Examples
    --------

    >>> map = plot_24h_rainfall(postcodes)
    """
    t = tool.Tool()
    lat_long = t.get_lat_long(postcodes)
    rain_per_hour = []
    for num in range(len(lat_long)):
        rain = live.get_rainfall_per_hour(READING_FILE, lat_long.latitude[num],
                                          lat_long.longitude[num])
        rain_per_hour.append(rain)

    td = pd.date_range('20210507', periods=24, freq='H')
    td = pd.DataFrame(pd.to_datetime(td))

    dataframe = pd.DataFrame()
    for i in range(len(postcodes)):
        la = lat_long.latitude[i]
        lon = lat_long.longitude[i]

        latitude = pd.DataFrame(pd.Series(la))
        longitude = pd.DataFrame(pd.Series(lon))
        latmul = pd.concat([latitude] * 24, ignore_index=True)
        longmul = pd.concat([longitude] * 24, ignore_index=True)
        if rain_per_hour[i] == 'n':
            rain_per_hour[i] = np.zeros(24)
        a = pd.Series(rain_per_hour[i])
        rain = pd.DataFrame(pd.concat([td, latmul, longmul, a], axis=1))
        dataframe = pd.concat([dataframe, rain], axis=0)

    dataframe.columns = ['DateIndex', 'latitude', 'longitude', 'value']
    dataframe['time'] = dataframe['DateIndex'].dt.time
    dataframe['value'].fillna(0, inplace=True)
    # locations = dataframe[['time', 'latitude', 'longitude',
    #                        'value']].values.tolist()

    graph_data = []
    for time in dataframe['time'].unique().tolist():
        graph_data.append(dataframe[dataframe['time'] == time][[
            'latitude', 'longitude', 'value'
        ]].values.tolist())
    # graph_data

    rain_map = folium.Map(location=[54, -1],
                          zoom_start=6,
                          tiles='OpenStreetMap')

    folium.plugins.HeatMapWithTime(
        graph_data,
        radius=10,
        gradient={
            .01: 'green',
            .2: 'white',
            .65: 'yellow',
            1: 'red'
        },
        auto_play=True,
    ).add_to(rain_map)

    return rain_map
