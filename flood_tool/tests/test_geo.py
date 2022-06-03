"""Test geo transformation module."""

import numpy as np

from pytest import mark, approx
from flood_tool import geo


def test_rad():
    """
    Test the function that convert degrees/minutes/seconds
    into radians.
    """

    assert np.isclose(geo.rad(60), np.pi/3.).all()
    assert np.isclose(geo.rad(90), np.pi/2.).all()
    assert np.isclose(geo.rad(2, 30), np.pi/72.).all()
    assert np.isclose(geo.rad(10, 120), np.pi/15.).all()
    assert np.isclose(geo.rad(10, 60, 3600), np.pi/15.).all()
    
def test_deg():
    """
    Test the function that convert degrees into radians.
    """

    assert np.isclose(geo.deg(np.pi/6), 30.0).all()
    assert np.isclose(geo.deg(np.pi/3), 60.0).all()
    assert np.isclose(geo.deg(np.pi/30., True), (6,0,0)).all()

def test_lat_long_to_xyz():
    """
    Test the function that convert latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.
    """

    latitude = np.array([geo.rad(18,69,22.7239)])
    longitude = np.array([geo.rad(8,35, 9.7263)])
    xyz = np.array([[5959064.3904905], [899738.16596551],[2079570.06221127]])

    datum = geo.osgb36
    datum.F_0=1.0

    assert np.isclose(geo.lat_long_to_xyz(latitude, longitude, True,
                               datum = datum), xyz).all()

def test_xyz_to_lat_long():
    '''
    Test the function that convert Cartesian (x,y,z) coordinates into
    latitude and longitude in a given datum.
    '''

    x = np.array([5959064.3904905])
    y = np.array([899738.16596551])
    z = np.array([2079570.06221127])

    latitude = np.array([geo.rad(18,69,22.7239)])
    longitude = np.array([geo.rad(8,35, 9.7263)])
    
    assert np.isclose(np.array(geo.xyz_to_lat_long(x, y, z, True)), 
                                    np.array((latitude, longitude))).all()

def test_WGS84toOSGB36():
    '''
    Test the function that convert WGS84 lat/long to OSGB36 lat/long.
    '''
    lat_long_wgs = np.array([[geo.rad(52, 39, 28.71)],
                             [geo.rad(1, 42, 57.79)]])
    lat_long_os = np.array([[geo.rad(52, 39, 27.2531)],
                            [geo.rad(1, 43, 4.5177)]])
    
    assert np.isclose(geo.WGS84toOSGB36(*lat_long_wgs,True), lat_long_os).all()

def test_OSGB36toWGS84():
    '''
    Test the function that convert OSGB36 lat/long to WGS84 lat/long.
    '''
    lat_long_wgs = np.array([[geo.rad(52, 39, 28.71)],
                             [geo.rad(1, 42, 57.79)]])
    lat_long_os = np.array([[geo.rad(52, 39, 27.2531)],
                            [geo.rad(1, 43, 4.5177)]])
    
    assert np.isclose(geo.OSGB36toWGS84(*lat_long_os,True), lat_long_wgs).all()

def test_WGS84transform():
    '''
    Test the WGS84transform
    '''
    xyz_wgs = np.array([[2374269.743895],
                        [ 227812.43485],
                        [697123.34544]])

    xyz_os = np.array([[2374667.4422326],
                        [227691.79941624],
                        [697648.4332051]])

    assert np.isclose(geo.WGS84transform(xyz_wgs), xyz_os).all()

def test_OSGB36transform():
    '''
    Test the OSGB36transform
    '''
    xyz_wgs = np.array([[2374269.743895],
                        [ 227812.43485],
                        [697123.34544]])

    xyz_os = np.array([[2374667.4422326],
                        [227691.79941624],
                        [697648.4332051]])

    assert np.isclose(geo.OSGB36transform(xyz_os), xyz_wgs).all()

def test_get_easting_northing_from_gps_lat_long():
    '''
    Test the get OSGB36 easting/northing from GPS latitude and
    longitude pairs.
    '''

    easting_northing_1 = (np.array([429169.12281358]), np.array([623305.43536952]))
    easting_northing_2 = (np.array([426652.6126664]), np.array([4160740.46143352]))

    assert np.isclose(geo.get_easting_northing_from_gps_lat_long([55.5], [-1.54]), easting_northing_1).all()
    assert np.isclose(geo.get_easting_northing_from_gps_lat_long([87.2], [2.87]), easting_northing_2).all()

def test_get_gps_lat_long_from_easting_northing():
    '''
    Test the get GPS latitude and longitude pairs from 
    OSGB36 easting/northing.
    '''

    lat_long_1 = (np.array([55.49733826]), np.array([-1.54022209]))
    lat_long_2 = (np.array([56.47803778]), np.array([0.11256682]))

    assert np.isclose(geo.get_gps_lat_long_from_easting_northing([429157], [623009]), lat_long_1).all()
    assert np.isclose(geo.get_gps_lat_long_from_easting_northing([530268], [734110]), lat_long_2).all()

if __name__ == "__main__":
    test_rad()