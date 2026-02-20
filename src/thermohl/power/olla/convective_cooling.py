# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

from thermohl import floatArrayLike
from thermohl.power.convective_cooling import ConvectiveCoolingBase
from thermohl.power.olla import Air


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term."""

    def __init__(
        self,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        ambient_temperature: floatArrayLike,
        wind_speed: floatArrayLike,
        wind_angle: floatArrayLike,
        outer_diameter: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            altitude (float | numpy.ndarray): Altitude (m).
            azimuth (float | numpy.ndarray): Azimuth (deg).
            ambient_temperature (float | numpy.ndarray): Ambient temperature (°C).
            wind_speed (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wind_angle (float | numpy.ndarray): Wind angle regarding north (deg).
            outer_diameter (float | numpy.ndarray): External diameter (m).

        """
        super().__init__(
            altitude,
            azimuth,
            ambient_temperature,
            wind_speed,
            wind_angle,
            outer_diameter,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
            **kwargs,
        )
