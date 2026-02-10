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

    def __init__(
        self,
        transit: floatArrayLike,
        magnetic_coeff: floatArrayLike,
        temperature_coeff_linear: floatArrayLike,
        dc_resistance_20c_ohm_m: floatArrayLike,
        reference_temperature_c: floatArrayLike = 20.0,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            transit (float | numpy.ndarray): Transit intensity (A).
            magnetic_coeff (float | numpy.ndarray): Coefficient for magnetic effects (—).
            temperature_coeff_linear (float | numpy.ndarray): Linear resistance augmentation with temperature (K⁻¹).
            dc_resistance_20c_ohm_m (float | numpy.ndarray): Electric resistance per unit length (DC) at 20°C (Ω·m⁻¹).
            reference_temperature_c (float | numpy.ndarray, optional): Reference temperature (°C). The default is 20.

        """
        self.current_a = transit
        self.magnetic_coeff = magnetic_coeff
        self.temp_coeff_linear = temperature_coeff_linear
        self.dc_resistance_20c = dc_resistance_20c_ohm_m
        self.reference_temp_c = reference_temperature_c

    def value(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            conductor_temp_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return (
            self.magnetic_coeff
            * self.dc_resistance_20c
            * (
                1.0
                + self.temp_coeff_linear * (conductor_temp_c - self.reference_temp_c)
            )
            * self.current_a**2
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term derivative (W·m⁻¹·K⁻¹).

        """
        return (
            self.magnetic_coeff
            * self.dc_resistance_20c
            * self.temp_coeff_linear
            * self.current_a**2
            * np.ones_like(conductor_temperature)
        )
