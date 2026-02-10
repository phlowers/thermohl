# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun
from thermohl.power.power_term import PowerTerm


class _SRad:
    """Solar radiation calculator."""

    def __init__(self, clean: List[float], indus: List[float]):
        """Initialize the solar radiation calculator.

        Args:
            clean (list[float]): Coefficients for the polynomial function to compute atmospheric turbidity in clean air conditions.
            indus (list[float]): Coefficients for the polynomial function to compute atmospheric turbidity in industrial (polluted) air conditions.
        """
        self.clean = clean
        self.indus = indus

    def catm(
        self, x: floatArrayLike, trb: Optional[floatArrayLike] = 0.0
    ) -> floatArrayLike:
        """Compute coefficient for atmosphere turbidity.
        This method calculates the atmospheric turbidity coefficient using a polynomial
        function of the solar altitude. The coefficients of the polynomial are a weighted
        average of the clean air and industrial air coefficients, with the weights
        determined by the turbidity factor.

        Args:
            x (float | numpy.ndarray): Solar altitude in degrees.
            trb (float | numpy.ndarray): Atmospheric turbidity factor (0 for clean air, 1 for industrial air).

        Returns:
            float | numpy.ndarray: Coefficient for atmospheric turbidity.
        """
        clean_weight = 1.0 - trb
        coeff_6 = clean_weight * self.clean[6] + trb * self.indus[6]
        coeff_5 = clean_weight * self.clean[5] + trb * self.indus[5]
        coeff_4 = clean_weight * self.clean[4] + trb * self.indus[4]
        coeff_3 = clean_weight * self.clean[3] + trb * self.indus[3]
        coeff_2 = clean_weight * self.clean[2] + trb * self.indus[2]
        coeff_1 = clean_weight * self.clean[1] + trb * self.indus[1]
        coeff_0 = clean_weight * self.clean[0] + trb * self.indus[0]
        return (
            coeff_6 * x**6
            + coeff_5 * x**5
            + coeff_4 * x**4
            + coeff_3 * x**3
            + coeff_2 * x**2
            + coeff_1 * x**1
            + coeff_0
        )

    def __call__(
        self,
        lat: floatArrayLike,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        trb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        solar_altitude_rad = sun.solar_altitude(lat, month, day, hour)
        solar_azimuth_rad = sun.solar_azimuth(lat, month, day, hour)
        incidence_angle_rad = np.arccos(
            np.cos(solar_altitude_rad) * np.cos(solar_azimuth_rad - azimuth)
        )
        altitude_factor = 1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude**2
        clearness_factor = self.catm(np.rad2deg(solar_altitude_rad), trb)
        solar_irradiance = (
            altitude_factor * clearness_factor * np.sin(incidence_angle_rad)
        )
        return np.where(solar_irradiance > 0.0, solar_irradiance, 0.0)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
        self,
        lat: floatArrayLike,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        tb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        outer_diameter_m: floatArrayLike,
        alpha: floatArrayLike,
        est: _SRad,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        self.solar_absorptivity = alpha
        if srad is None:
            self.solar_irradiance = est(
                np.deg2rad(lat),
                altitude,
                np.deg2rad(azimuth),
                tb,
                month,
                day,
                hour,
            )
        else:
            self.solar_irradiance = np.maximum(srad, 0.0)
        self.outer_diameter_m = outer_diameter_m

    def value(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        r"""Compute solar heating.

        Args:
            conductor_temp_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return (
            self.solar_absorptivity
            * self.solar_irradiance
            * self.outer_diameter_m
            * np.ones_like(conductor_temp_c)
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(conductor_temperature)
