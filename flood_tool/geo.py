from numpy import array, asarray, mod, sin, cos, tan, sqrt, arctan2, \
    floor, rad2deg, deg2rad, stack
from scipy.linalg import inv

__all__ = ['get_easting_northing_from_gps_lat_long',
           'get_gps_lat_long_from_easting_northing']


class Ellipsoid(object):
    """Class to hold Ellipsoid information."""

    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a-b)/(a+b)
        self.e2 = (a**2-b**2)/a**2
        self.F_0 = F_0
        self.H = 0


class Datum(Ellipsoid):
    """Class to hold datum information."""

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H


def rad(deg, min=0, sec=0):
    """Convert degrees/minutes/seconds into radians.

    Parameters
    ----------

    deg: float/arraylike
       Value(s) in degrees
    min: float/arraylike
       Value(s) in minutes
    sec: float/arraylike
       Value(s) in (angular) seconds

    Returns
    -------
    numpy.ndarray
         Equivalent values in radians
    """
    deg = asarray(deg)
    min = asarray(min)
    sec = asarray(sec)
    return deg2rad(deg+min/60.+sec/3600.)


def deg(rad, dms=False):
    """Convert degrees into radians.

    Parameters
    ----------

    deg: float/arraylike
        Value(s) in degrees

    Returns
    -------
    np.ndarray
        Equivalent values in radians.
    """
    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0*mod(deg, 1.0)
        sec = 60.0*mod(min, 1.0)
        return stack((floor(deg),  floor(min), sec.round(4)))
    else:
        return deg


# data for OSGB36 lat/long datum.
osgb36 = Datum(a=6377563.396,
               b=6356256.910,
               F_0=0.9996012717,
               phi_0=rad(49.0),
               lam_0=rad(-2.),
               E_0=400000,
               N_0=-100000,
               H=24.7)

# data for WGS84 GPS datum.
wgs84 = Ellipsoid(a=6378137,
                  b=6356752.3142,
                  F_0=0.9996)


def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    """Convert latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.
    """
    if not rads:
        phi = rad(phi)
        lam = rad(lam)

    nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)

    return array(((nu+datum.H)*cos(phi)*cos(lam),
                  (nu+datum.H)*cos(phi)*sin(lam),
                  ((1-datum.e2)*nu+datum.H)*sin(phi)))


def xyz_to_lat_long(x, y, z, rads=False, datum=osgb36):
    """Convert Cartesian (x,y,z) coordinates into
    latitude and longitude in a given datum.
    """

    p = sqrt(x**2+y**2)

    lam = arctan2(y, x)
    phi = arctan2(z, p*(1-datum.e2))

    for _ in range(10):

        nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
        dnu = (-datum.a*datum.F_0*cos(phi)*sin(phi)
               / (1-datum.e2*sin(phi)**2)**1.5)

        f0 = (z + datum.e2*nu*sin(phi))/p - tan(phi)
        f1 = datum.e2*(nu**cos(phi)+dnu*sin(phi))/p - 1.0/cos(phi)**2
        phi -= f0/f1

    if not rads:
        phi = deg(phi)
        lam = deg(lam)

    return phi, lam


def get_easting_northing_from_gps_lat_long(phi, lam, rads=False):
    """ Get OSGB36 easting/northing from GPS latitude and
    longitude pairs.

    Parameters
    ----------

    phi: float/arraylike
        GPS (i.e. WGS84 datum) latitude value(s)
    lam: float/arraylike
        GPS (i.e. WGS84 datum) longitude value(s).
    rads: bool (optional)
        If true, specifies input is is radians.

    Returns
    -------
    numpy.ndarray
        Easting values (in m)
    numpy.ndarray
        Northing values (in m)

    Examples
    --------

    >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54])
    (array([429157.0]), array([623009]))

    References
    ----------

    Based on the formulas in "A guide to coordinate systems in Great Britain".

    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """
    phi, lam = WGS84toOSGB36(phi, lam, rads)
    phi = rad(phi)
    lam = rad(lam)
    datum = osgb36

    s_phi = sin(phi)
    c_phi = cos(phi)
    # equations from the Ordnance Survey website
    v = datum.a*datum.F_0*(1 - datum.e2*s_phi**2)**(-0.5)
    rho = datum.a*datum.F_0*(1 - datum.e2)*(1 - datum.e2*s_phi**2)**(-1.5)
    ita_square = v/rho - 1

    # components of a long equation calculating M
    phi_add = phi + datum.phi_0
    phi_diff = phi - datum.phi_0
    sum1 = (1 + datum.n + 1.25*datum.n**2 + 1.25*datum.n**3)*phi_diff
    sum2 = (3*datum.n + 3*datum.n**2 +
            21/8*datum.n**3)*sin(phi_diff)*cos(phi_add)
    sum3 = (15/8*datum.n**2 + 15/8*datum.n**3)*sin(2*phi_diff)*cos(2*phi_add)
    sum4 = 35/24*datum.n**3*sin(3*phi_diff)*cos(3*phi_add)

    M = datum.b*datum.F_0*(sum1 - sum2 + sum3 - sum4)
    I_i = M + datum.N_0

    tan_square = tan(phi)**2
    lam_diff = lam - datum.lam_0

    II = v/2 * s_phi*c_phi
    III = v/24 * s_phi*c_phi**3*(5 - tan_square + 9*ita_square)
    IIIA = v/720 * s_phi*c_phi**5*(61 - 58*tan_square + tan_square**2)
    IV = v*c_phi
    V = v/6 * c_phi**3*(v/rho - tan_square)
    VI = v/120 * c_phi**5*(5 - 18*tan_square + tan_square**2 + 14*ita_square
                           - 58*tan_square*ita_square)

    north_os = I_i + II*lam_diff**2 + III*lam_diff**4 + IIIA*lam_diff**6
    east_os = datum.E_0 + IV*lam_diff + V*lam_diff**3 + VI*lam_diff**5

    return east_os, north_os


def get_gps_lat_long_from_easting_northing(east, north,
                                           rads=False, dms=False):
    """ Get OSGB36 easting/northing from GPS latitude and
    longitude pairs.

    Parameters
    ----------

    east: float/arraylike
        OSGB36 easting value(s) (in m).
    north: float/arraylike
        OSGB36 easting value(s) (in m).
    rads: bool (optional)
        If true, specifies ouput is is radians.
    dms: bool (optional)
        If true, output is in degrees/minutes/seconds. Incompatible
        with rads option.

    Returns
    -------
    numpy.ndarray
        GPS (i.e. WGS84 datum) latitude value(s).
    numpy.ndarray
        GPS (i.e. WGS84 datum) longitude value(s).

    Examples
    --------

    >>> get_gps_lat_long_from_easting_northing([429157], [623009])
    (array([55.5]), array([-1.540008]))

    References
    ----------

    Based on the formulas in "A guide to coordinate systems in Great Britain".

    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """

    datum = osgb36
    north = array(north)
    east = array(east)
    phi_diff = (north - datum.N_0)/(datum.a*datum.F_0) + datum.phi_0

    poly1 = 1 + datum.n + 5 / 4 * datum.n ** 2 + 5 / 4 * datum.n ** 3
    poly2 = 3 * datum.n + 3 * datum.n ** 2 + 21 / 8 * datum.n ** 3
    poly3 = 15 / 8 * datum.n ** 2 + 15 / 8 * datum.n ** 3
    poly4 = 35 / 24 * datum.n ** 3

    M = datum.b * datum.F_0 * (
        poly1 * (phi_diff - datum.phi_0)
        - poly2 * sin(phi_diff - datum.phi_0) * cos(phi_diff + datum.phi_0)
        + poly3 * sin(2 * (phi_diff - datum.phi_0)) *
        cos(2 * (phi_diff + datum.phi_0))
        - poly4 * sin(3 * (phi_diff - datum.phi_0)) *
        cos(3 * (phi_diff + datum.phi_0))
    )

    # while abs(north - datum.N_0 - M) >= 0.01:
    length = len(north)
    for i in range(length):
        while abs(north[i] - datum.N_0 - M[i]) >= 0.01:
            phi_diff_new = (north[i] - datum.N_0 - M[i]) / \
                (datum.a*datum.F_0) + phi_diff[i]
            M[i] = datum.b * datum.F_0 * (
                poly1 * (phi_diff_new - datum.phi_0)
                - poly2 * sin(phi_diff_new - datum.phi_0) *
                cos(phi_diff_new + datum.phi_0)
                + poly3 * sin(2 * (phi_diff_new - datum.phi_0)) *
                cos(2 * (phi_diff_new + datum.phi_0))
                - poly4 * sin(3 * (phi_diff_new - datum.phi_0)) *
                cos(3 * (phi_diff_new + datum.phi_0))
            )
            phi_diff[i] = phi_diff_new

    rho = datum.a * datum.F_0 * (1 - datum.e2) * \
        ((1 - datum.e2 * (sin(phi_diff)) ** 2) ** -1.5)
    v = datum.a * datum.F_0 * (1 - datum.e2 * (sin(phi_diff)) ** 2) ** -0.5
    ita2 = v/rho - 1

    tan_phi_diff = tan(phi_diff)
    VII = tan_phi_diff/(2*rho*v)
    VIII = tan_phi_diff/(24*rho*v**3)*(5+3*tan_phi_diff **
                                       2+ita2-9*tan_phi_diff**2*ita2)
    IX = tan_phi_diff*(61 + 90*tan_phi_diff**2 + 45 *
                       tan_phi_diff ** 4)/(720*rho*v**5)
    X = 1/cos(phi_diff)/v
    XI = (v/rho + 2*tan_phi_diff**2) / (cos(phi_diff) * 6*v**3)

    XII_numerator = 5 + 28*tan_phi_diff**2 + 24*tan_phi_diff**4
    XII_denominator = cos(phi_diff) * 120 * v**5
    XII = XII_numerator/XII_denominator

    XIIA_numerator = 61 + 622*tan_phi_diff**2 + \
        1320*tan_phi_diff**4 + 720*tan_phi_diff**6
    XIIA_denominator = cos(phi_diff) * 5040 * v**7
    XIIA = XIIA_numerator/XIIA_denominator

    phi = phi_diff - VII*(east-datum.E_0)**2 + \
        VIII*(east-datum.E_0)**4 - IX*(east-datum.E_0)**6
    lamda = datum.lam_0 + X*(east-datum.E_0) - XI*(east-datum.E_0)**3 + \
        XII*(east-datum.E_0)**5 - XIIA*(east-datum.E_0)**7

    gps_latitude, gps_longitude = OSGB36toWGS84(deg(phi), deg(lamda), False)

    if rads is True:
        return rad(gps_latitude), rad(gps_longitude)

    if dms is True:
        return deg(rad(gps_latitude), dms=True), deg(
            rad(gps_longitude),
            dms=True
        )

    return gps_latitude, gps_longitude


class HelmertTransform(object):
    """Class to perform a Helmert Transform."""

    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))

        self.M = array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]])

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.T + self.M@X


class HelmertInverseTransform(object):
    """Class to perform the inverse of a Helmert Transform."""

    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))

        self.M = inv(array([[1+s, -rz, ry],
                            [rz, 1+s, -rx],
                            [-ry, rx, 1+s]]))

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.M@(X-self.T)


OSGB36transform = HelmertTransform(20.4894e-6,
                                   -rad(0, 0, 0.1502),
                                   -rad(0, 0, 0.2470),
                                   -rad(0, 0, 0.8421),
                                   array([-446.448, 125.157, -542.060]))

WGS84transform = HelmertInverseTransform(20.4894e-6,
                                         -rad(0, 0, 0.1502),
                                         -rad(0, 0, 0.2470),
                                         -rad(0, 0, 0.8421),
                                         array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(lat, long, rads=False):
    """Convert WGS84 lat/long to OSGB36 lat/long."""
    X = OSGB36transform(lat_long_to_xyz(asarray(lat), asarray(long),
                                        rads=rads, datum=wgs84))
    return xyz_to_lat_long(*X, rads=rads, datum=osgb36)


def OSGB36toWGS84(lat, long, rads=False):
    """Convert OSGB36 lat/long to WGS84 lat/long."""
    X = WGS84transform(lat_long_to_xyz(asarray(lat), asarray(long),
                                       rads=rads, datum=osgb36))
    return xyz_to_lat_long(*X, rads=rads, datum=wgs84)
