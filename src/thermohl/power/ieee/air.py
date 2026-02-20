# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from thermohl import floatArrayLike


class Air:
    """Air quantities."""

    @staticmethod
    def volumic_mass(
        air_temperature: floatArrayLike, altitude: floatArrayLike = 0.0
    ) -> floatArrayLike:
        r"""Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (°C).
            altitude (float | numpy.ndarray, optional): Altitude above sea level (m). The default is 0.

        Returns:
            float | numpy.ndarray: Volumic mass in kg·m⁻³.

        """
        return (1.293 - 1.525e-04 * altitude + 6.379e-09 * altitude**2) / (
            1.0 + 0.00367 * air_temperature
        )

    @staticmethod
    def dynamic_viscosity(air_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute air dynamic viscosity.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (°C)

        Returns:
            float | numpy.ndarray: Dynamic viscosity in kg·m⁻¹·s⁻¹.

        """
        return (1.458e-06 * (air_temperature + 273.0) ** 1.5) / (
            air_temperature + 383.4
        )

    @staticmethod
    def thermal_conductivity(air_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute air thermal conductivity.

        Args:
            air_temperature (float | numpy.ndarray): Air temperature (°C)

        Returns:
            float | numpy.ndarray: Thermal conductivity in W·m⁻¹·K⁻¹.

        """
        return 2.424e-02 + 7.477e-05 * air_temperature - 4.407e-09 * air_temperature**2
