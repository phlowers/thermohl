# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl import floatArrayLike

_zerok = 273.15


def kelvin(t: floatArrayLike) -> floatArrayLike:
    return t + _zerok


class Air:
    """`Wikipedia <https://fr.wikipedia.org/wiki/Air> models."""

    @staticmethod
    def volumic_mass(
        air_temperature: floatArrayLike, altitude: floatArrayLike = 0.0
    ) -> floatArrayLike:
        r"""
        Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (in Celsius).
            altitude (float | numpy.ndarray, optional): Altitude above sea-level. The default is 0.

        Returns:
            float | numpy.ndarray: Volumic mass in kg·m⁻³.

        """
        air_temperature_k = kelvin(air_temperature)
        return (
            1.292
            * _zerok
            * np.exp(-3.42e-02 * altitude / air_temperature_k)
            / air_temperature_k
        )

    @staticmethod
    def dynamic_viscosity(air_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute air dynamic viscosity.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (in Celsius)

        Returns:
            float | numpy.ndarray: Dynamic viscosity in kg·m⁻¹·s⁻¹.

        """
        air_temperature_k = kelvin(air_temperature)
        return (
            8.8848e-15 * air_temperature_k**3
            - 3.2398e-11 * air_temperature_k**2
            + 6.2657e-08 * air_temperature_k
            + 2.3543e-06
        )

    @staticmethod
    def kinematic_viscosity(
        air_temperature: floatArrayLike, altitude: floatArrayLike = 0.0
    ) -> floatArrayLike:
        r"""Compute air kinematic viscosity.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (in Celsius)
            altitude (float | numpy.ndarray, optional): Altitude above sea-level. The default is 0.

        Returns:
            float | numpy.ndarray: Kinematic viscosity in m²·s⁻¹.

        """
        return Air.dynamic_viscosity(air_temperature) / Air.volumic_mass(
            air_temperature, altitude=altitude
        )

    @staticmethod
    def thermal_conductivity(air_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute air thermal conductivity.

        The output is valid for input in [-150, 1300] range (in Celsius)

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (in Celsius)

        Returns:
            float | numpy.ndarray: Thermal conductivity in W·m⁻¹·K⁻¹.

        """
        air_temperature_k = kelvin(air_temperature)
        return (
            1.5207e-11 * air_temperature_k**3
            - 4.8570e-08 * air_temperature_k**2
            + 1.0184e-04 * air_temperature_k
            - 3.9333e-04
        )
