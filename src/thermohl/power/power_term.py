# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC
from typing import Any

import numpy as np

from thermohl import floatArrayLike

_dT = 1.0e-03


class PowerTerm(ABC):
    """Base class for power term."""

    def __init__(self, **kwargs: Any):
        pass

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute power term value in function of temperature.

        Usually this function should be overridden in children classes; if it is
        not the case it will just return zero.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return np.zeros_like(T) if not np.isscalar(T) else 0.0

    def derivative(
        self, conductor_temperature: floatArrayLike, dT: float = _dT
    ) -> floatArrayLike:
        r"""Compute power term derivative regarding temperature in function of temperature.

        Usually this function should be overriden in children classes; if it is
        not the case it will evaluate the derivative from the value method with
        a second-order approximation.

        Parameters
        ----------
        conductor_temperature : float or np.ndarray
            Conductor temperature (C).
        dT : float, optional
            Temperature increment. The default is 1.0E-03.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return (
            self.value(conductor_temperature + dT)
            - self.value(conductor_temperature - dT)
        ) / (2.0 * dT)
