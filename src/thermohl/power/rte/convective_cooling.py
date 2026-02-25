# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power.rte import Air
from thermohl.power.convective_cooling import ConvectiveCoolingBase


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term.

    Very similar to IEEE. The differences are in some coefficient values for air
    constants.
    """

    def __init__(
        self,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        ambient_temperature: floatArrayLike,
        wind_speed: floatArrayLike,
        wind_azimuth: floatArrayLike,
        outer_diameter: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            altitude (float | numpy.ndarray): Altitude (m).
            cable_azimuth (float | numpy.ndarray): Azimuth (deg).
            ambient_temperature (float | numpy.ndarray): Ambient temperature (°C).
            wind_speed (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wind_azimuth (float | numpy.ndarray): wind_azimuth regarding north (deg).
            outer_diameter (float | numpy.ndarray): External diameter (m).

        """
        super().__init__(
            altitude,
            cable_azimuth,
            ambient_temperature,
            wind_speed,
            wind_azimuth,
            outer_diameter,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
        )

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Parameters
        ----------
        conductor_temperature : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        film_temperature = 0.5 * (conductor_temperature + self.ambient_temp)
        temperature_delta = conductor_temperature - self.ambient_temp
        # very slight difference with air.IEEE.volumic_mass() in coefficient before altitude**2
        air_density = (
            1.293 - 1.525e-04 * self.altitude + 6.38e-09 * self.altitude**2
        ) / (1 + 0.00367 * film_temperature)
        return np.maximum(
            self._value_forced(film_temperature, temperature_delta, air_density),
            self._value_natural(temperature_delta, air_density),
        )
