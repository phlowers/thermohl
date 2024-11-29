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

    def _celsius2kelvin(self, T: floatArrayLike) -> floatArrayLike:
        return T + self.zerok

    def __init__(
        self,
        Ta: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
        sigma: float = 5.67e-08,
        zerok: float = 273.15,
        **kwargs: Any,
    ):
        r"""Init with args.

        Parameters
        ----------
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.
        zerok : float, optional
            Value for zero kelvin.

        Returns
        -------

        """
        self.zerok = zerok
        self.Ta = self._celsius2kelvin(Ta)
        self.D = D
        self.epsilon = epsilon
        self.sigma = sigma

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return (
            np.pi
            * self.sigma
            * self.epsilon
            * self.D
            * (self._celsius2kelvin(T) ** 4 - self.Ta**4)
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Analytical derivative of value method.

        Parameters
        ----------
        conductor_temperature : float or np.ndarray
        Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return (
            4.0 * np.pi * self.sigma * self.epsilon * self.D * conductor_temperature**3
        )
