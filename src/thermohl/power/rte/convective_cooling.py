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
        azimuth: floatArrayLike,
        ambient_temperature_c: floatArrayLike,
        wind_speed_ms: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            altitude (float | numpy.ndarray): Altitude (m).
            azimuth (float | numpy.ndarray): Azimuth (deg).
            ambient_temperature_c (float | numpy.ndarray): Ambient temperature (°C).
            wind_speed_ms (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wa (float | numpy.ndarray): Wind angle regarding north (deg).
            D (float | numpy.ndarray): External diameter (m).

        """
        super().__init__(
            altitude,
            azimuth,
            ambient_temperature_c,
            wind_speed_ms,
            wa,
            D,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
        )

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        film_temp_c = 0.5 * (T + self.ambient_temp_c)
        temp_delta_c = T - self.ambient_temp_c
        # very slight difference with air.IEEE.volumic_mass() in coefficient before altitude**2
        air_density = (
            1.293 - 1.525e-04 * self.altitude_m + 6.38e-09 * self.altitude_m**2
        ) / (1 + 0.00367 * film_temp_c)
        return np.maximum(
            self._value_forced(film_temp_c, temp_delta_c, air_density),
            self._value_natural(temp_delta_c, air_density),
        )
