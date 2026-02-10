# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Callable, Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power import PowerTerm


class ConvectiveCoolingBase(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        rho: Callable[[floatArrayLike, floatArrayLike], floatArrayLike],
        mu: Callable[[floatArrayLike], floatArrayLike],
        lambda_: Callable[[floatArrayLike], floatArrayLike],
        **kwargs: Any,
    ):
        self.altitude_m = altitude
        self.ambient_temp_c = Ta
        self.wind_speed_ms = ws
        self.attack_angle_rad = np.arcsin(
            np.sin(np.deg2rad(np.abs(azimuth - wa) % 180.0))
        )
        self.outer_diameter_m = D

        self.air_density = rho
        self.dynamic_viscosity = mu
        self.thermal_conductivity = lambda_

    def _value_forced(
        self,
        film_temp_c: floatArrayLike,
        temp_delta_c: floatArrayLike,
        air_density: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute forced convective cooling value.

        Args:
            film_temp_c (float | numpy.ndarray): Film temperature (°C).
            temp_delta_c (float | numpy.ndarray): Temperature difference (°C).
            air_density (float | numpy.ndarray): Velocity magnitude proxy (relative density or similar, model-dependent).

        Returns:
            float | numpy.ndarray: Computed forced convective cooling values (W·m⁻¹).
        """
        reynolds = (
            self.wind_speed_ms
            * self.outer_diameter_m
            * air_density
            / self.dynamic_viscosity(film_temp_c)
        )
        direction_factor = (
            1.194
            - np.cos(self.attack_angle_rad)
            + 0.194 * np.cos(2.0 * self.attack_angle_rad)
            + 0.368 * np.sin(2.0 * self.attack_angle_rad)
        )
        return (
            direction_factor
            * np.maximum(1.01 + 1.35 * reynolds**0.52, 0.754 * reynolds**0.6)
            * self.thermal_conductivity(film_temp_c)
            * temp_delta_c
        )

    def _value_natural(
        self,
        temp_delta_c: floatArrayLike,
        air_density: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute natural convective cooling value.

        Args:
            temp_delta_c (float | numpy.ndarray): Temperature difference (°C).
            air_density (float | numpy.ndarray): Velocity magnitude (relative density or similar, model-dependent).

        Returns:
            float | numpy.ndarray: Natural convective cooling value (W·m⁻¹).
        """
        return (
            3.645
            * np.sqrt(air_density)
            * self.outer_diameter_m**0.75
            * np.sign(temp_delta_c)
            * np.abs(temp_delta_c) ** 1.25
        )

    def value(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Args:
            conductor_temp_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        film_temp_c = 0.5 * (conductor_temp_c + self.ambient_temp_c)
        temp_delta_c = conductor_temp_c - self.ambient_temp_c
        air_density = self.air_density(film_temp_c, self.altitude_m)
        return np.maximum(
            self._value_forced(film_temp_c, temp_delta_c, air_density),
            self._value_natural(temp_delta_c, air_density),
        )
