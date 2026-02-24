# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Utility functions to compute different sun positions from a point on earth.

These positions usually depend on the point latitude and the time. The sun
position is then used to estimate the solar radiation in CIGRE and IEEE
models.
"""
from math import pi

import numpy as np
from thermohl import floatArrayLike, intArrayLike


def utc2solar_hour(
    utc_hour: floatArrayLike,
    day_of_month: intArrayLike,
    month_index: intArrayLike,
    longitude: floatArrayLike,
) -> floatArrayLike:
    """convert utc hour to solar hour adding the longitude contribution

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    utc_hour : float or numpy.ndarray
        Hour of the day (must be between 0 and 24 excluded).
    longitude : float or numpy.ndarray, optional
        Longitude (in rad)

    Returns
    -------
    float or numpy.ndarray
        solar hour

    """
    day_of_year = _csm[month_index - 1] + day_of_month
    B = 2 * pi * (day_of_year - 81) / 365
    solar_hour = (
        utc_hour
        + longitude / (2 * pi) * 24
        - (7.678 * np.sin(B + 1.374) - 9.87 * np.sin(2 * B)) / 60
    )

    return solar_hour


def hour_angle(
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
) -> floatArrayLike:
    """Compute hour angle.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    solar_hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    solar_minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    solar_second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Hour angle in radians.

    """
    solar_hour = solar_hour % 24 + solar_minute / 60.0 + solar_second / 3600.0
    return np.radians(15.0 * (solar_hour - 12.0))


_csm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def solar_declination(
    month_index: intArrayLike, day_of_month: intArrayLike
) -> floatArrayLike:
    """Compute solar declination.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    month_index : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day_of_month: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    Returns
    -------
    float or numpy.ndarray
        Solar declination in radians.

    """
    day_of_year = _csm[month_index - 1] + day_of_month
    return np.deg2rad(23.46) * np.sin(2.0 * np.pi * (day_of_year + 284) / 365.0)


def solar_altitude(
    latitude: floatArrayLike,
    month_index: intArrayLike,
    day_of_month: intArrayLike,
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
) -> floatArrayLike:
    """Compute solar altitude.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    latitude : float or numpy.ndarray
        latitude in radians.
    month_index : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day_of_month: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    solar_hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    solar_minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    solar_second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Solar altitude in radians.

    """
    computed_solar_declination = solar_declination(month_index, day_of_month)
    computed_hour_angle = hour_angle(solar_hour, solar_minute, solar_second)
    return np.arcsin(
        np.cos(latitude)
        * np.cos(computed_solar_declination)
        * np.cos(computed_hour_angle)
        + np.sin(latitude) * np.sin(computed_solar_declination)
    )


def solar_azimuth(
    latitude: floatArrayLike,
    month_index: intArrayLike,
    day_of_month: intArrayLike,
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
) -> floatArrayLike:
    """Compute solar azimuth.

    If more than one input are numpy arrays, they should have the same size.

    Parameters
    ----------
    latitude : float or numpy.ndarray
        latitude in radians.
    month_index : int or numpy.ndarray
        Month number (must be between 1 and 12)
    day_of_month: int or numpy.ndarray
        Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
        month)
    solar_hour : float or numpy.ndarray
        Hour of the day (solar, must be between 0 and 23).
    solar_minute : float or numpy.ndarray, optional
        Minutes on the clock. The default is 0.
    solar_second : float or numpy.ndarray, optional
        Seconds on the clock. The default is 0.

    Returns
    -------
    float or numpy.ndarray
        Solar azimuth in radians.

    """
    computed_solar_declination = solar_declination(month_index, day_of_month)
    computed_hour_angle = hour_angle(solar_hour, solar_minute, solar_second)
    azimuth_ratio = np.sin(computed_hour_angle) / (
        np.sin(latitude) * np.cos(computed_hour_angle)
        - np.cos(latitude) * np.tan(computed_solar_declination)
    )
    azimuth_offset_rad = np.where(
        azimuth_ratio >= 0.0,
        np.where(computed_hour_angle < 0.0, 0.0, np.pi),
        np.where(computed_hour_angle < 0.0, np.pi, 2.0 * np.pi),
    )
    return azimuth_offset_rad + np.arctan(azimuth_ratio)
