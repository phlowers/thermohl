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
        ambient_temperature: floatArrayLike,
        wind_speed: floatArrayLike,
        wind_angle: floatArrayLike,
        outer_diameter: floatArrayLike,
        air_density: Callable[[floatArrayLike, floatArrayLike], floatArrayLike],
        dynamic_viscosity: Callable[[floatArrayLike], floatArrayLike],
        thermal_conductivity: Callable[[floatArrayLike], floatArrayLike],
        **kwargs: Any,
    ):
        self.altitude = altitude
        self.ambient_temp = ambient_temperature
        self.wind_speed = wind_speed
        self.attack_angle = np.arcsin(
            np.sin(np.deg2rad(np.abs(azimuth - wind_angle) % 180.0))
        )
        self.outer_diameter = outer_diameter

        self.air_density = air_density
        self.dynamic_viscosity = dynamic_viscosity
        self.thermal_conductivity = thermal_conductivity

    def _value_forced(
        self,
        film_temperature: floatArrayLike,
        temperature_delta: floatArrayLike,
        air_density: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute forced convective cooling value.

        Args:
            film_temperature (float | numpy.ndarray): Film temperature (°C).
            temperature_delta (float | numpy.ndarray): Temperature difference (°C).
            air_density (float | numpy.ndarray): Velocity magnitude proxy (relative density or similar, model-dependent).

        Returns:
            float | numpy.ndarray: Computed forced convective cooling values (W·m⁻¹).
        """
        reynolds = (
            self.wind_speed
            * self.outer_diameter
            * air_density
            / self.dynamic_viscosity(film_temperature)
        )
        direction_factor = (
            1.194
            - np.cos(self.attack_angle)
            + 0.194 * np.cos(2.0 * self.attack_angle)
            + 0.368 * np.sin(2.0 * self.attack_angle)
        )
        return (
            direction_factor
            * np.maximum(1.01 + 1.35 * reynolds**0.52, 0.754 * reynolds**0.6)
            * self.thermal_conductivity(film_temperature)
            * temperature_delta
        )

    def _value_natural(
        self,
        temperature_delta: floatArrayLike,
        air_density: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute natural convective cooling value.

        Args:
            temperature_delta (float | numpy.ndarray): Temperature difference (°C).
            air_density (float | numpy.ndarray): Velocity magnitude (relative density or similar, model-dependent).

        Returns:
            float | numpy.ndarray: Natural convective cooling value (W·m⁻¹).
        """
        return (
            3.645
            * np.sqrt(air_density)
            * self.outer_diameter**0.75
            * np.sign(temperature_delta)
            * np.abs(temperature_delta) ** 1.25
        )

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        film_temperature = 0.5 * (conductor_temperature + self.ambient_temp)
        temperature_delta = conductor_temperature - self.ambient_temp
        air_density = self.air_density(film_temperature, self.altitude)
        return np.maximum(
            self._value_forced(film_temperature, temperature_delta, air_density),
            self._value_natural(temperature_delta, air_density),
        )
