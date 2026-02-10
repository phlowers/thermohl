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
        TLow: floatArrayLike,
        THigh: floatArrayLike,
        linear_resistance_temp_low_ohm_m: floatArrayLike,
        RDCHigh: floatArrayLike,
    ) -> floatArrayLike:
        return (RDCHigh - linear_resistance_temp_low_ohm_m) / (THigh - TLow)

    def __init__(
        self,
        transit: floatArrayLike,
        TLow: floatArrayLike,
        THigh: floatArrayLike,
        linear_resistance_temp_low_ohm_m: floatArrayLike,
        RDCHigh: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            transit (float | numpy.ndarray): Transit intensity (A).
            TLow (float | numpy.ndarray): Temperature for linear_resistance_temp_low_ohm_m measurement (°C).
            THigh (float | numpy.ndarray): Temperature for RDCHigh measurement (°C).
            linear_resistance_temp_low_ohm_m (float | numpy.ndarray): Electric resistance per unit length at TLow (Ω·m⁻¹).
            RDCHigh (float | numpy.ndarray): Electric resistance per unit length at THigh (Ω·m⁻¹).

        """
        self.temp_low_c = TLow
        self.temp_high_c = THigh
        self.dc_resistance_low_c = linear_resistance_temp_low_ohm_m
        self.dc_resistance_high_c = RDCHigh
        self.current_a = transit
        self.temp_coeff_linear = JouleHeating._c(
            TLow, THigh, linear_resistance_temp_low_ohm_m, RDCHigh
        )

    def _rdc(self, conductor_temperature_c: floatArrayLike) -> floatArrayLike:
        return self.dc_resistance_low_c + self.temp_coeff_linear * (
            conductor_temperature_c - self.temp_low_c
        )

    def value(self, conductor_temperature_c: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            conductor_temperature_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return self._rdc(conductor_temperature_c) * self.current_a**2

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
            * self.current_a**2
            * np.ones_like(conductor_temperature)
        )
