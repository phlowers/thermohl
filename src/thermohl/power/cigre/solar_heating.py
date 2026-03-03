# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any, Iterable

import numpy as np

from thermohl import floatArrayLike, sun as sun, datetimeListLike
from thermohl.power import PowerTerm
from thermohl.sun import time_to_float_hours


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _solar_radiation(
        latitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        albedo: floatArrayLike,
        datetime_utc: datetimeListLike,
    ) -> floatArrayLike:
        """Compute solar radiation.

        :param latitude: Latitude in radians.
        :param cable_azimuth: Azimuth of the conductor in radians.
        :param albedo: Albedo.
        :param datetime_utc: Datetime in UTC.
        :return: Solar radiation.
        """
        date = (
            np.array([d.date() for d in datetime_utc])
            if isinstance(datetime_utc, Iterable)
            else datetime_utc.date()
        )
        hour = (
            np.array([time_to_float_hours(d.time()) for d in datetime_utc])
            if isinstance(datetime_utc, Iterable)
            else time_to_float_hours(datetime_utc.time())
        )
        solar_declination_rad = sun.solar_declination(date)
        hour_angle_rad = sun.hour_angle(hour)
        solar_altitude_rad = sun.solar_altitude(latitude, date, hour)
        direct_irradiance = (
            1280.0 * np.sin(solar_altitude_rad) / (0.314 + np.sin(solar_altitude_rad))
        )
        solar_azimuth_rad = np.arcsin(
            np.cos(solar_declination_rad)
            * np.sin(hour_angle_rad)
            / np.cos(solar_altitude_rad)
        )
        incidence_angle_rad = np.arccos(
            np.cos(solar_altitude_rad) * np.cos(solar_azimuth_rad - cable_azimuth)
        )
        direct_term = 0.5 * np.pi * albedo * np.sin(solar_altitude_rad) + np.sin(
            incidence_angle_rad
        )
        sin_altitude = np.sin(solar_altitude_rad)
        clear_sky_factor = np.piecewise(
            sin_altitude,
            [sin_altitude < 0.0, sin_altitude >= 0.0],
            [lambda value: 0.0, lambda value: value**1.2],
        )
        diffuse_term = (
            0.5
            * np.pi
            * (1 + albedo)
            * (570.0 - 0.47 * direct_irradiance)
            * clear_sky_factor
        )
        return np.where(
            solar_altitude_rad > 0.0,
            direct_term * direct_irradiance + diffuse_term,
            0.0,
        )

    def __init__(
        self,
        latitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        albedo: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        measured_solar_irradiance: floatArrayLike,
        **kwargs: Any,
    ):
        """Init with args.
        If more than one input are numpy arrays, they should have the same size.

        :param latitude: Latitude in degrees.
        :param cable_azimuth: Azimuth of the conductor in degrees.
        :param albedo: Albedo.
        :param datetime_utc: Datetime in UTC.
        :param outer_diameter: external diameter of the conductor.
        :param solar_absorptivity: Solar absorption coefficient of the conductor.
        :param measured_solar_irradiance: Optional precomputed solar radiation term.
        """
        self.solar_absorptivity = solar_absorptivity

        mask = np.isnan(measured_solar_irradiance)
        self.solar_irradiance = np.empty_like(measured_solar_irradiance)
        if np.any(~mask):
            self.solar_irradiance[~mask] = np.maximum(measured_solar_irradiance, 0.0)
        if np.any(mask):
            self.solar_irradiance[mask] = SolarHeating._solar_radiation(
                np.deg2rad(latitude),
                np.deg2rad(cable_azimuth),
                albedo,
                datetime_utc,
            )

        self.outer_diameter = outer_diameter

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating.
        If more than one input are numpy arrays, they should have the same size.

        :param conductor_temperature: Conductor temperature (°C).
        :return: Power term value (W·m⁻¹).
        """
        return (
            self.solar_absorptivity
            * self.solar_irradiance
            * self.outer_diameter
            * np.ones_like(conductor_temperature)
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(conductor_temperature)
