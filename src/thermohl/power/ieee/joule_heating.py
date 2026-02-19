# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power import PowerTerm


class JouleHeating(PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _c(
        temp_low: floatArrayLike,
        temp_high: floatArrayLike,
        linear_resistance_temp_low: floatArrayLike,
        linear_resistance_temp_high: floatArrayLike,
    ) -> floatArrayLike:
        return (linear_resistance_temp_high - linear_resistance_temp_low) / (
            temp_high - temp_low
        )

    def __init__(
        self,
        transit: floatArrayLike,
        temp_low: floatArrayLike,
        temp_high: floatArrayLike,
        linear_resistance_temp_low: floatArrayLike,
        linear_resistance_temp_high: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            transit (float | numpy.ndarray): Transit intensity (A).
            temp_low (float | numpy.ndarray): Temperature for linear_resistance_temp_low measurement (°C).
            temp_high (float | numpy.ndarray): Temperature for linear_resistance_temp_high measurement (°C).
            linear_resistance_temp_low (float | numpy.ndarray): Electric resistance per unit length at temp_low (Ω·m⁻¹).
            linear_resistance_temp_high (float | numpy.ndarray): Electric resistance per unit length at temp_high (Ω·m⁻¹).

        """
        self.temp_low = temp_low
        self.temp_high = temp_high
        self.linear_resistance_temp_low = linear_resistance_temp_low
        self.linear_resistance_temp_high = linear_resistance_temp_high
        self.transit = transit
        self.temp_coeff_linear = JouleHeating._c(
            temp_low,
            temp_high,
            linear_resistance_temp_low,
            linear_resistance_temp_high,
        )

    def _rdc(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        return self.linear_resistance_temp_low + self.temp_coeff_linear * (
            conductor_temperature - self.temp_low
        )

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return self._rdc(conductor_temperature) * self.transit**2

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term derivative (W·m⁻¹·K⁻¹).

        """
        return (
            self.temp_coeff_linear
            * self.transit**2
            * np.ones_like(conductor_temperature)
        )
