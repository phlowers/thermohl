# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun as sun
from thermohl.power import PowerTerm


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _solar_radiation(
        latitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        albedo: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        solar_declination_rad = sun.solar_declination(month, day)
        hour_angle_rad = sun.hour_angle(hour)
        solar_altitude_rad = sun.solar_altitude(latitude, month, day, hour)
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
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        precomputed_solar_radiation: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            latitude (float | numpy.ndarray): Latitude.
            cable_azimuth (float | numpy.ndarray): Azimuth.
            albedo (float | numpy.ndarray): Albedo.
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (solar, must be between 0 and 23).
            outer_diameter (float | numpy.ndarray): external diameter.
            solar_absorptivity (float | numpy.ndarray): Solar absorption coefficient.
            precomputed_solar_radiation (float | numpy.ndarray | None): Optional precomputed solar radiation term.
        """
        self.solar_absorptivity = solar_absorptivity
        if precomputed_solar_radiation is None:
            self.solar_irradiance = SolarHeating._solar_radiation(
                np.deg2rad(latitude),
                np.deg2rad(cable_azimuth),
                albedo,
                month,
                day,
                hour,
            )
        else:
            self.solar_irradiance = precomputed_solar_radiation
        self.outer_diameter = outer_diameter

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute solar heating.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

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
