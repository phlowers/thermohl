# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Generic radiative cooling term."""

from typing import Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power.power_term import PowerTerm


class RadiativeCoolingBase(PowerTerm):
    """Generic power term for radiative cooling."""

    def _celsius2kelvin(self, temperature: floatArrayLike) -> floatArrayLike:
        return temperature + self.kelvin_offset

    def __init__(
        self,
        ambient_temperature: floatArrayLike,
        outer_diameter: floatArrayLike,
        emissivity: floatArrayLike,
        stefan_boltzmann_constant: float = 5.67e-08,
        zerok: float = 273.15,
        **kwargs: Any,
    ):
        r"""Init with args.

        Args:
            ambient_temperature (float | numpy.ndarray): Ambient temperature (°C).
            outer_diameter (float | numpy.ndarray): External diameter (m).
            emissivity (float | numpy.ndarray): Emissivity (—).
            stefan_boltzmann_constant (float, optional): Stefan–Boltzmann constant (W·m⁻²·K⁻⁴). The default is 5.67e-08.
            zerok (float, optional): Offset to convert Celsius to Kelvin (K). The default is 273.15.

        """
        self.kelvin_offset = zerok
        self.ambient_temperature = self._celsius2kelvin(ambient_temperature)
        self.outer_diameter = outer_diameter
        self.emissivity = emissivity
        self.stefan_boltzmann = stefan_boltzmann_constant

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return (
            np.pi
            * self.stefan_boltzmann
            * self.emissivity
            * self.outer_diameter
            * (
                self._celsius2kelvin(conductor_temperature) ** 4
                - self.ambient_temperature**4
            )
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Analytical derivative of value method.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (K).

        Returns:
            float | numpy.ndarray: Power term derivative (W·m⁻¹·K⁻¹).

        """
        return (
            4.0
            * np.pi
            * self.stefan_boltzmann
            * self.emissivity
            * self.outer_diameter
            * conductor_temperature**3
        )
