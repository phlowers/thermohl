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
        I: floatArrayLike,
        D: floatArrayLike,
        d: floatArrayLike,
        A: floatArrayLike,
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
            I (float | numpy.ndarray): Transit intensity (A).
            D (float | numpy.ndarray): External diameter (m).
            d (float | numpy.ndarray): Core diameter (m).
            A (float | numpy.ndarray): External (total) cross-sectional area (m²).
            a (float | numpy.ndarray): Core cross-sectional area (m²).
            km (float | numpy.ndarray): Coefficient for magnetic effects (—).
            ki (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).
            kl (float | numpy.ndarray): Linear resistance augmentation with temperature (K⁻¹).
            kq (float | numpy.ndarray): Quadratic resistance augmentation with temperature (K⁻²).
            RDC20 (float | numpy.ndarray): Electric resistance per unit length (DC) at 20°C (Ω·m⁻¹).
            T20 (float | numpy.ndarray, optional): Reference temperature (°C). The default is 20.
            f (float | numpy.ndarray, optional): Current frequency (Hz). The default is 50.

        """
        self.I = I
        self.D = D
        self.d = d
        self.kem = self._kem(A, a, km, ki)
        self.kl = kl
        self.kq = kq
        self.RDC20 = RDC20
        self.T20 = T20
        self.f = f

    def _rdc(self, T: floatArrayLike) -> floatArrayLike:
        """
        Compute resistance per unit length for direct current.

        Args:
            T (float | numpy.ndarray): Temperature at which to compute the resistance (°C).

        Returns:
            float | numpy.ndarray: Resistance per unit length for direct current at the given temperature(s) (Ω·m⁻¹).
        """
        dt = T - self.T20
        return self.RDC20 * (1.0 + self.kl * dt + self.kq * dt**2)

    def _ks(self, rdc: floatArrayLike) -> floatArrayLike:
        """
        Compute skin-effect coefficient.

        This method calculates the skin-effect coefficient based on the given
        resistance (rdc) and the object's attributes. The calculation is an
        approximation as described in the RTE's document.

        Args:
            rdc (float | numpy.ndarray): The resistance value(s) for which the skin-effect coefficient is to be computed (Ω·m⁻¹).

        Returns:
            floatArrayLike: The computed skin-effect coefficient(s) (—).
        """
        z = (
            8
            * np.pi
            * self.f
            * (self.D - self.d) ** 2
            / ((self.D**2 - self.d**2) * 1.0e07 * rdc)
        )
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1.0 - self.d / self.D
        return 1.0 + a * (1.0 - 0.5 * beta - b * beta**2)

    def _kem(
        self,
        A: floatArrayLike,
        a: floatArrayLike,
        km: floatArrayLike,
        ki: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute magnetic coefficient.

        Args:
            A (float | numpy.ndarray): External (total) cross-sectional area (m²).
            a (float | numpy.ndarray): Core cross-sectional area (m²).
            km (float | numpy.ndarray): Coefficient for magnetic effects (—).
            ki (float | numpy.ndarray): Coefficient for magnetic effects (A⁻¹).

        Returns:
            floatArrayLike: Computed magnetic coefficient (—).
        """
        s = (
            np.ones_like(self.I)
            * np.ones_like(A)
            * np.ones_like(a)
            * np.ones_like(km)
            * np.ones_like(ki)
        )
        z = s.shape == ()
        if z:
            s = np.array([1.0])
        I_ = self.I * s
        a_ = a * s
        A_ = A * s
        m = a_ > 0.0
        ki_ = ki * s
        kem = km * s
        kem[m] += ki_[m] * I_[m] / ((A_[m] - a_[m]) * 1.0e06)
        if z:
            kem = kem[0]
        return kem

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            T (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        rdc = self._rdc(T)
        ks = self._ks(rdc)
        rac = self.kem * ks * rdc
        return rac * self.I**2
