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
        km: floatArrayLike,
        kl: floatArrayLike,
        RDC20: floatArrayLike,
        T20: floatArrayLike = 20.0,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            I (float | numpy.ndarray): Transit intensity (A).
            km (float | numpy.ndarray): Coefficient for magnetic effects (—).
            kl (float | numpy.ndarray): Linear resistance augmentation with temperature (K⁻¹).
            RDC20 (float | numpy.ndarray): Electric resistance per unit length (DC) at 20°C (Ω·m⁻¹).
            T20 (float | numpy.ndarray, optional): Reference temperature (°C). The default is 20.

        """
        self.I = I
        self.km = km
        self.kl = kl
        self.RDC20 = RDC20
        self.T20 = T20

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Args:
            T (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return self.km * self.RDC20 * (1.0 + self.kl * (T - self.T20)) * self.I**2

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term derivative (W·m⁻¹·K⁻¹).

        """
        return (
            self.km
            * self.RDC20
            * self.kl
            * self.I**2
            * np.ones_like(conductor_temperature)
        )
