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
from datetime import time
from math import pi
import numpy as np
from thermohl import (
    floatArrayLike,
    dateListLike,
    datetimeListLike,
)


def utc2solar_hour(
    datetime_utc: datetimeListLike,
    longitude: floatArrayLike,
) -> floatArrayLike:
    """Convert UTC datetime to solar hour adding the longitude contribution.
    If more than one input are numpy arrays, they should have the same size.

    :param datetime_utc: Datetime in UTC.
    :param longitude: Longitude (in rad).
    :return: Solar hour.
    """

    def time_to_float_hours(t: time) -> float:
        return t.hour + t.minute / 60 + t.second / 3600

    day_of_year = np.array([d.timetuple().tm_yday for d in datetime_utc])
    utc_hour = np.array([time_to_float_hours(d.time()) for d in datetime_utc])
    B = 2 * pi * (day_of_year - 81) / 365
    solar_hour = (
        utc_hour
        + longitude / (2 * pi) * 24
        - (7.678 * np.sin(B + 1.374) - 9.87 * np.sin(2 * B)) / 60
    )
    return solar_hour


def hour_angle(solar_hour: floatArrayLike) -> floatArrayLike:
    """Compute hour angle.
    If more than one input are numpy arrays, they should have the same size.

    :param solar_hour: solar hour of the day.
    :return: Hour angle in radians.
    """
    return np.radians(15.0 * (solar_hour - 12.0))


def solar_declination(date: dateListLike) -> floatArrayLike:
    """Compute solar declination.
    If more than one input are numpy arrays, they should have the same size.

    :param date: Date of the year.
    :return: Solar declination in radians.
    """
    day_of_year = np.array([d.timetuple().tm_yday for d in date])
    return np.deg2rad(23.46) * np.sin(2.0 * np.pi * (day_of_year + 284) / 365.0)


def solar_altitude(
    latitude: floatArrayLike,
    date: dateListLike,
    solar_hour: floatArrayLike,
) -> floatArrayLike:
    """Compute solar altitude.
    If more than one input are numpy arrays, they should have the same size.

    :param latitude: latitude in radians.
    :param date: Date of the year.
    :param solar_hour: solar hour of the day.
    :return: Solar altitude in radians.
    """
    computed_solar_declination = solar_declination(date)
    computed_hour_angle = hour_angle(solar_hour)
    return np.arcsin(
        np.cos(latitude)
        * np.cos(computed_solar_declination)
        * np.cos(computed_hour_angle)
        + np.sin(latitude) * np.sin(computed_solar_declination)
    )


def solar_azimuth(
    latitude: floatArrayLike,
    date: dateListLike,
    solar_hour: floatArrayLike,
) -> floatArrayLike:
    """Compute solar azimuth.
    If more than one input are numpy arrays, they should have the same size.

    :param latitude: latitude in radians.
    :param date: Date of the year.
    :param solar_hour: solar hour of the day.
    :return: Solar azimuth in radians.
    """
    computed_solar_declination = solar_declination(date)
    computed_hour_angle = hour_angle(solar_hour)
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
