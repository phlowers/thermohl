# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Callable, Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power import PowerTerm


logger = logging.getLogger(__name__)


def compute_wind_attack_angle(
    cable_azimuth: floatArrayLike, wind_azimuth: floatArrayLike
) -> floatArrayLike:
    """
    Compute wind attack angle.

    Args:
        cable_azimuth (float | numpy.ndarray): Cable azimuth (deg).
        wind_azimuth (float | numpy.ndarray): Wind azimuth regarding north (deg).

    Returns:
        float | numpy.ndarray: Wind attack angle (rad).
    """
    return np.arcsin(np.sin(np.deg2rad(np.abs(cable_azimuth - wind_azimuth) % 180.0)))


class ConvectiveCoolingBase(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        ambient_temperature: floatArrayLike,
        wind_speed: floatArrayLike,
        outer_diameter: floatArrayLike,
        air_density: Callable[[floatArrayLike, floatArrayLike], floatArrayLike],
        dynamic_viscosity: Callable[[floatArrayLike], floatArrayLike],
        thermal_conductivity: Callable[[floatArrayLike], floatArrayLike],
        wind_azimuth: floatArrayLike = None,
        wind_attack_angle: floatArrayLike = None,
        **kwargs: Any,
    ):
        self._check_arguments(wind_azimuth, wind_attack_angle)
        self._set_wind_attack_angle(cable_azimuth, wind_azimuth, wind_attack_angle)

        self.altitude = altitude
        self.ambient_temp = ambient_temperature
        self.wind_speed = wind_speed

        self.outer_diameter = outer_diameter

        self.air_density = air_density
        self.dynamic_viscosity = dynamic_viscosity
        self.thermal_conductivity = thermal_conductivity

    def _check_arguments(
        self, wind_azimuth: floatArrayLike, wind_attack_angle: floatArrayLike
    ) -> None:
        if (
            wind_attack_angle is None or np.isnan(wind_attack_angle).any()
        ) and wind_azimuth is None:
            raise ValueError("Must provide either wind_attack_angle or wind_azimuth.")
        if (
            wind_attack_angle is not None
            and not np.isnan(wind_attack_angle).all()
            and wind_azimuth is not None
        ):
            logger.warning(
                "Both wind_attack_angle and wind_azimuth are provided. wind_azimuth will be ignored."
            )

    def _set_wind_attack_angle(
        self,
        cable_azimuth: floatArrayLike,
        wind_azimuth: floatArrayLike,
        wind_attack_angle: floatArrayLike,
    ) -> None:
        # Compute missing wind attack angles
        if isinstance(wind_attack_angle, np.ndarray) and wind_attack_angle.ndim > 0:
            mask = np.isnan(wind_attack_angle)
            if np.any(mask):
                wind_attack_angle[mask] = compute_wind_attack_angle(
                    cable_azimuth[mask], wind_azimuth[mask]
                )
            self.wind_attack_angle = wind_attack_angle
        elif wind_attack_angle is None or np.isnan(wind_attack_angle):
            self.wind_attack_angle = compute_wind_attack_angle(
                cable_azimuth, wind_azimuth
            )
        else:
            self.wind_attack_angle = wind_attack_angle

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
            - np.cos(self.wind_attack_angle)
            + 0.194 * np.cos(2.0 * self.wind_attack_angle)
            + 0.368 * np.sin(2.0 * self.wind_attack_angle)
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
