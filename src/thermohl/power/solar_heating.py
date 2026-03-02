# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, Optional, Any

import numpy as np

from thermohl import floatArrayLike, sun, datetimeListLike
from thermohl.power.power_term import PowerTerm
from thermohl.sun import time_to_float_hours


class _SRad:
    """Solar radiation calculator."""

    def __init__(self, clean: List[float], indus: List[float]):
        """Initialize the solar radiation calculator.

        :param clean: Coefficients for the polynomial function to compute atmospheric turbidity in clean air conditions.
        :param indus: Coefficients for the polynomial function to compute atmospheric turbidity in industrial (polluted) air conditions.
        """
        self.clean = clean
        self.indus = indus

    def atmosphere_turbidity(
        self,
        solar_altitude: floatArrayLike,
        turbidity: Optional[floatArrayLike] = 0.0,
    ) -> floatArrayLike:
        """Compute coefficient for atmosphere turbidity.
        This method calculates the atmospheric turbidity coefficient using a polynomial
        function of the solar altitude. The coefficients of the polynomial are a weighted
        average of the clean air and industrial air coefficients, with the weights
        determined by the turbidity factor.

        :param solar_altitude: Solar altitude in degrees.
        :param turbidity: Atmospheric turbidity factor (0 for clean air, 1 for industrial air).
        :return: Coefficient for atmospheric turbidity.
        """
        clean_weight = 1.0 - turbidity
        coeff_6 = clean_weight * self.clean[6] + turbidity * self.indus[6]
        coeff_5 = clean_weight * self.clean[5] + turbidity * self.indus[5]
        coeff_4 = clean_weight * self.clean[4] + turbidity * self.indus[4]
        coeff_3 = clean_weight * self.clean[3] + turbidity * self.indus[3]
        coeff_2 = clean_weight * self.clean[2] + turbidity * self.indus[2]
        coeff_1 = clean_weight * self.clean[1] + turbidity * self.indus[1]
        coeff_0 = clean_weight * self.clean[0] + turbidity * self.indus[0]
        return (
            coeff_6 * solar_altitude**6
            + coeff_5 * solar_altitude**5
            + coeff_4 * solar_altitude**4
            + coeff_3 * solar_altitude**3
            + coeff_2 * solar_altitude**2
            + coeff_1 * solar_altitude**1
            + coeff_0
        )

    def __call__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        turbidity: floatArrayLike,
        datetime_utc: datetimeListLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        date = [d.date() for d in datetime_utc]
        hour = np.array([time_to_float_hours(d.time()) for d in datetime_utc])
        computed_solar_altitude = sun.solar_altitude(latitude, date, hour)
        computed_solar_azimuth = sun.solar_azimuth(latitude, date, hour)
        computed_incidence_angle = np.arccos(
            np.cos(computed_solar_altitude)
            * np.cos(computed_solar_azimuth - cable_azimuth)
        )
        altitude_factor = 1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude**2
        clearness_factor = self.atmosphere_turbidity(
            np.rad2deg(computed_solar_altitude), turbidity
        )
        solar_irradiance = (
            altitude_factor * clearness_factor * np.sin(computed_incidence_angle)
        )
        return np.where(solar_irradiance > 0.0, solar_irradiance, 0.0)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        turbidity: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        est: _SRad,
        measured_solar_irradiance: floatArrayLike,
        **kwargs: Any,
    ):
        self.solar_absorptivity = solar_absorptivity

        mask = np.isnan(measured_solar_irradiance)
        self.solar_irradiance = np.empty_like(measured_solar_irradiance)
        if np.any(~mask):
            self.solar_irradiance[~mask] = np.maximum(measured_solar_irradiance, 0.0)
        if np.any(mask):
            self.solar_irradiance[mask] = est(
                np.deg2rad(latitude),
                altitude,
                np.deg2rad(cable_azimuth),
                turbidity,
                datetime_utc,
            )

        self.outer_diameter = outer_diameter

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating.

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
        """Compute solar heating derivative.

        :param conductor_temperature: Conductor temperature.
        :return: Derivative of solar heating.
        """
        return np.zeros_like(conductor_temperature)
