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
        outer_diameter: floatArrayLike,
        core_diameter: floatArrayLike,
        outer_area: floatArrayLike,
        core_area: floatArrayLike,
        magnetic_coeff: floatArrayLike,
        magnetic_coeff_per_a: floatArrayLike,
        temperature_coeff_linear: floatArrayLike,
        temperature_coeff_quadratic: floatArrayLike,
        linear_resistance_dc_20c: floatArrayLike,
        reference_temperature: floatArrayLike = 20.0,
        frequency: floatArrayLike = 50.0,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            transit (float | numpy.ndarray): Transit intensity (A).
            outer_diameter (float | numpy.ndarray): External diameter (m).
            core_diameter (float | numpy.ndarray): Core diameter (m).
            outer_area (float | numpy.ndarray): External (total) cross-sectional area (m²).
            core_area (float | numpy.ndarray): Core cross-sectional area (m²).
            magnetic_coeff (float | numpy.ndarray): Coefficient for magnetic effects (—).
            magnetic_coeff_per_a (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).
            temperature_coeff_linear (float | numpy.ndarray): Linear resistance augmentation with temperature (K⁻¹).
            temperature_coeff_quadratic (float | numpy.ndarray): Quadratic resistance augmentation with temperature (K⁻²).
            linear_resistance_dc_20c (float | numpy.ndarray): Electric resistance per unit length (DC) at 20°C (Ω·m⁻¹).
            reference_temperature (float | numpy.ndarray, optional): Reference temperature (°C). The default is 20.
            frequency (float | numpy.ndarray, optional): Current frequency (Hz). The default is 50.

        """
        self.transit = transit
        self.outer_diameter = outer_diameter
        self.core_diameter = core_diameter
        self.magnetic_coeff = self._kem(
            outer_area, core_area, magnetic_coeff, magnetic_coeff_per_a
        )
        self.temp_coeff_linear = temperature_coeff_linear
        self.temp_coeff_quadratic = temperature_coeff_quadratic
        self.linear_resistance_dc_20c = linear_resistance_dc_20c
        self.reference_temperature = reference_temperature
        self.frequency = frequency

    def _rdc(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """
        Compute resistance per unit length for direct current.

        Args:
            conductor_temperature (float | numpy.ndarray): Temperature at which to compute the resistance (°C).

        Returns:
            float | numpy.ndarray: Resistance per unit length for direct current at the given temperature(s) (Ω·m⁻¹).
        """
        temperature_delta = conductor_temperature - self.reference_temperature
        return self.linear_resistance_dc_20c * (
            1.0
            + self.temp_coeff_linear * temperature_delta
            + self.temp_coeff_quadratic * temperature_delta**2
        )

    def _ks(self, dc_resistance: floatArrayLike) -> floatArrayLike:
        """
        Compute skin-effect coefficient.

        This method calculates the skin-effect coefficient based on the given
        resistance (rdc) and the object's attributes. The calculation is an
        approximation as described in the RTE's document.

        Args:
            dc_resistance (float | numpy.ndarray): The resistance value(s) for which the skin-effect coefficient is to be computed (Ω·m⁻¹).

        Returns:
            floatArrayLike: The computed skin-effect coefficient(s) (—).
        """
        skin_param = (
            8
            * np.pi
            * self.frequency
            * (self.outer_diameter - self.core_diameter) ** 2
            / (
                (self.outer_diameter**2 - self.core_diameter**2)
                * 1.0e07
                * dc_resistance
            )
        )
        coeff_a = 7 * skin_param**2 / (315 + 3 * skin_param**2)
        coeff_b = 56 / (211 + skin_param**2)
        core_ratio = 1.0 - self.core_diameter / self.outer_diameter
        return 1.0 + coeff_a * (1.0 - 0.5 * core_ratio - coeff_b * core_ratio**2)

    def _kem(
        self,
        outer_area: floatArrayLike,
        core_area: floatArrayLike,
        magnetic_coeff: floatArrayLike,
        magnetic_coeff_per_a: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute magnetic coefficient.

        Args:
            outer_area (float | numpy.ndarray): External (total) cross-sectional area (m²).
            core_area (float | numpy.ndarray): Core cross-sectional area (m²).
            magnetic_coeff (float | numpy.ndarray): Coefficient for magnetic effects (—).
            magnetic_coeff_per_a (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).

        Returns:
            floatArrayLike: Computed magnetic coefficient (—).
        """
        scale = (
            np.ones_like(self.transit)
            * np.ones_like(outer_area)
            * np.ones_like(core_area)
            * np.ones_like(magnetic_coeff)
            * np.ones_like(magnetic_coeff_per_a)
        )
        is_scalar = scale.shape == ()
        if is_scalar:
            scale = np.array([1.0])
        current = self.transit * scale
        core_area = core_area * scale
        outer_area = outer_area * scale
        has_core = core_area > 0.0
        magnetic_slope = magnetic_coeff_per_a * scale
        magnetic_coeff = magnetic_coeff * scale
        magnetic_coeff[has_core] += (
            magnetic_slope[has_core]
            * current[has_core]
            / ((outer_area[has_core] - core_area[has_core]) * 1.0e06)
        )
        if is_scalar:
            magnetic_coeff = magnetic_coeff[0]
        return magnetic_coeff

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        dc_resistance = self._rdc(conductor_temperature)
        skin_effect_coeff = self._ks(dc_resistance)
        ac_resistance = self.magnetic_coeff * skin_effect_coeff * dc_resistance
        return ac_resistance * self.transit**2
