# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Callable, Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power import PowerTerm


class ConvectiveCoolingBase(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        rho: Callable[[floatArrayLike, floatArrayLike], floatArrayLike],
        mu: Callable[[floatArrayLike], floatArrayLike],
        lambda_: Callable[[floatArrayLike], floatArrayLike],
        **kwargs: Any,
    ):
        self.alt = alt
        self.Ta = Ta
        self.ws = ws
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.0)))
        self.D = D

        self.rho = rho
        self.mu = mu
        self.lambda_ = lambda_

    def _value_forced(
        self,
        Tf: floatArrayLike,
        Td: floatArrayLike,
        vm: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute forced convective cooling value.

        Parameters
        ----------
        Tf : float or np.ndarray
            Temperature field array.
        Td : float or np.ndarray
            Temperature difference array.
        vm : float or np.ndarray
            Velocity magnitude array.

        Returns
        -------
        float or np.ndarray
            Computed forced convective cooling values.
        """
        Re = self.ws * self.D * vm / self.mu(Tf)
        Kp = (
            1.194
            - np.cos(self.da)
            + 0.194 * np.cos(2.0 * self.da)
            + 0.368 * np.sin(2.0 * self.da)
        )
        return (
            Kp
            * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6)
            * self.lambda_(Tf)
            * Td
        )

    def _value_natural(
        self,
        Td: floatArrayLike,
        vm: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute natural convective cooling value.

        Parameters
        ----------
        Td : float or np.ndarray
            Temperature difference array.
        vm : float or np.ndarray
            Velocity magnitude array.

        Returns
        -------
        float or np.ndarray
            Natural convective cooling value array.
        """
        return 3.645 * np.sqrt(vm) * self.D**0.75 * np.sign(Td) * np.abs(Td) ** 1.25

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Tf = 0.5 * (T + self.Ta)
        Td = T - self.Ta
        vm = self.rho(Tf, self.alt)
        return np.maximum(self._value_forced(Tf, Td, vm), self._value_natural(Td, vm))
