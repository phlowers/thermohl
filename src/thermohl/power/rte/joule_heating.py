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
        outer_diameter_m: floatArrayLike,
        core_diameter_m: floatArrayLike,
        outer_area_m2: floatArrayLike,
        a: floatArrayLike,
        km: floatArrayLike,
        ki: floatArrayLike,
        kl: floatArrayLike,
        kq: floatArrayLike,
        RDC20: floatArrayLike,
        T20: floatArrayLike = 20.0,
        f: floatArrayLike = 50.0,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            transit (float | numpy.ndarray): Transit intensity (A).
            outer_diameter_m (float | numpy.ndarray): External diameter (m).
            core_diameter_m (float | numpy.ndarray): Core diameter (m).
            outer_area_m2 (float | numpy.ndarray): External (total) cross-sectional area (m²).
            a (float | numpy.ndarray): Core cross-sectional area (m²).
            km (float | numpy.ndarray): Coefficient for magnetic effects (—).
            ki (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).
            kl (float | numpy.ndarray): Linear resistance augmentation with temperature (K⁻¹).
            kq (float | numpy.ndarray): Quadratic resistance augmentation with temperature (K⁻²).
            RDC20 (float | numpy.ndarray): Electric resistance per unit length (DC) at 20°C (Ω·m⁻¹).
            T20 (float | numpy.ndarray, optional): Reference temperature (°C). The default is 20.
            f (float | numpy.ndarray, optional): Current frequency (Hz). The default is 50.

        """
        self.current_a = transit
        self.outer_diameter_m = outer_diameter_m
        self.core_diameter_m = core_diameter_m
        self.magnetic_coeff = self._kem(outer_area_m2, a, km, ki)
        self.temp_coeff_linear = kl
        self.temp_coeff_quadratic = kq
        self.dc_resistance_20c = RDC20
        self.reference_temp_c = T20
        self.frequency_hz = f

    def _rdc(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        """
        Compute resistance per unit length for direct current.

        Args:
            conductor_temp_c (float | numpy.ndarray): Temperature at which to compute the resistance (°C).

        Returns:
            float | numpy.ndarray: Resistance per unit length for direct current at the given temperature(s) (Ω·m⁻¹).
        """
        temp_delta_c = conductor_temp_c - self.reference_temp_c
        return self.dc_resistance_20c * (
            1.0
            + self.temp_coeff_linear * temp_delta_c
            + self.temp_coeff_quadratic * temp_delta_c**2
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
            * self.frequency_hz
            * (self.outer_diameter_m - self.core_diameter_m) ** 2
            / (
                (self.outer_diameter_m**2 - self.core_diameter_m**2)
                * 1.0e07
                * dc_resistance
            )
        )
        coeff_a = 7 * skin_param**2 / (315 + 3 * skin_param**2)
        coeff_b = 56 / (211 + skin_param**2)
        core_ratio = 1.0 - self.core_diameter_m / self.outer_diameter_m
        return 1.0 + coeff_a * (1.0 - 0.5 * core_ratio - coeff_b * core_ratio**2)

    def _kem(
        self,
        outer_area_m2: floatArrayLike,
        a: floatArrayLike,
        km: floatArrayLike,
        ki: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute magnetic coefficient.

        Args:
            outer_area_m2 (float | numpy.ndarray): External (total) cross-sectional area (m²).
            a (float | numpy.ndarray): Core cross-sectional area (m²).
            km (float | numpy.ndarray): Coefficient for magnetic effects (—).
            ki (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).

        Returns:
            floatArrayLike: Computed magnetic coefficient (—).
        """
        scale = (
            np.ones_like(self.current_a)
            * np.ones_like(outer_area_m2)
            * np.ones_like(a)
            * np.ones_like(km)
            * np.ones_like(ki)
        )
        is_scalar = scale.shape == ()
        if is_scalar:
            scale = np.array([1.0])
        current = self.current_a * scale
        core_area = a * scale
        outer_area = outer_area_m2 * scale
        has_core = core_area > 0.0
        magnetic_slope = ki * scale
        magnetic_coeff = km * scale
        magnetic_coeff[has_core] += (
            magnetic_slope[has_core]
            * current[has_core]
            / ((outer_area[has_core] - core_area[has_core]) * 1.0e06)
        )
        if is_scalar:
            magnetic_coeff = magnetic_coeff[0]
        return magnetic_coeff

    def value(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            conductor_temp_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        dc_resistance = self._rdc(conductor_temp_c)
        skin_effect_coeff = self._ks(dc_resistance)
        ac_resistance = self.magnetic_coeff * skin_effect_coeff * dc_resistance
        return ac_resistance * self.current_a**2
