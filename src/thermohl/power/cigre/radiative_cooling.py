# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

from thermohl import floatArrayLike
from thermohl.power import RadiativeCoolingBase


class RadiativeCooling(RadiativeCoolingBase):
    def __init__(
        self,
        ambient_temperature_c: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
        sigma: float = 5.67e-08,
        **kwargs: Any,
    ):
        r"""Init with args.

        Args:
            ambient_temperature_c (float | numpy.ndarray): Ambient temperature (°C).
            D (float | numpy.ndarray): External diameter (m).
            epsilon (float | numpy.ndarray): Emissivity (—).
            sigma (float, optional): Stefan–Boltzmann constant (W·m⁻²·K⁻⁴). The default is 5.67e-08.
        """
        super().__init__(
            ambient_temperature_c=ambient_temperature_c,
            D=D,
            epsilon=epsilon,
            sigma=sigma,
            zerok=273.0,
            **kwargs,
        )
