# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

from thermohl import floatArrayLike
from thermohl.power import PowerTerm


class RadiativeCooling(PowerTerm):
    """Power term for radiative cooling.

    Very similar to RadiativeCooling. Difference are in the
    Stefan-Boltzman constant value and the celsius-kelvin conversion.
    """

    def __init__(
        self,
        Ta: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
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

        """
        self.Ta = Ta
        self.D = D
        self.epsilon = epsilon

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
            17.8
            * self.epsilon
            * self.D
            * (((T + 273.0) / 100.0) ** 4 - ((self.Ta + 273.0) / 100.0) ** 4)
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
        return 4.0 * 1.78e-07 * self.epsilon * self.D * conductor_temperature**3
