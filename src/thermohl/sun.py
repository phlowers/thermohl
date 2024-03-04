"""Utility functions to compute different sun positions from a point on earth.

    These positions usually depend on the point latitude and the time. The sun
    position is then used to estimate the solar radiation in CIGRE and IEEE
    models.
"""
from typing import Union

import numpy as np


def hour_angle(hour: Union[float, np.ndarray],
               minute: Union[float, np.ndarray] = 0.,
               second: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
    """Compute hour angle.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Hour angle in radians.

    """
    solar_hour = hour % 24 + minute / 60. + second / 3600.
    return np.radians(15. * (solar_hour - 12.))


def solar_declination(month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]]):
    """Compute solar declination.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    month : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    Returns
    -------
    float or numpy.ndarray
        Solar declination in radians.

    """
    csm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    doy = csm[month - 1] + day
    return np.deg2rad(23.46) * np.sin(2. * np.pi * (doy + 284) / 365.)


def solar_altitude(lat: Union[float, np.ndarray],
                   month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]],
                   hour: Union[float, np.ndarray],
                   minute: Union[float, np.ndarray] = 0., second: Union[float, np.ndarray] = 0.) \
        -> Union[float, np.ndarray]:
    """Compute solar altitude.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    lat : float or numpy.ndarray
        latitude in radians.
    month : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Solar altitude in radians.

    """
    sd = solar_declination(month, day)
    ha = hour_angle(hour, minute=minute, second=second)
    return np.arcsin(np.cos(lat) * np.cos(sd) * np.cos(ha) + np.sin(lat) * np.sin(sd))


def solar_azimuth(lat: Union[float, np.ndarray],
                  month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]],
                  hour: Union[float, np.ndarray],
                  minute: Union[float, np.ndarray] = 0., second: Union[float, np.ndarray] = 0.) \
        -> Union[float, np.ndarray]:
    """Compute solar azimuth.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    lat : float or numpy.ndarray
        latitude in radians.
    month : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Solar azimuth in radians.

    """
    sd = solar_declination(month, day)
    ha = hour_angle(hour, minute=minute, second=second)
    Xi = np.sin(ha) / (np.sin(lat) * np.cos(ha) - np.cos(lat) * np.tan(sd))
    C = np.where(Xi >= 0.,
                 np.where(ha < 0., 0., np.pi),
                 np.where(ha < 0., np.pi, 2. * np.pi))
    return C + np.arctan(Xi)
