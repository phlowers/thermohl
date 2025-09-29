# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

from thermohl import floatArrayLike
from thermohl.power.convective_cooling import ConvectiveCoolingBase
from thermohl.power.olla import Air


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term."""

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            alt (float | numpy.ndarray): Altitude (m).
            azm (float | numpy.ndarray): Azimuth (deg).
            Ta (float | numpy.ndarray): Ambient temperature (°C).
            ws (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wa (float | numpy.ndarray): Wind angle regarding north (deg).
            D (float | numpy.ndarray): External diameter (m).

        """
        super().__init__(
            alt,
            azm,
            Ta,
            ws,
            wa,
            D,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
            **kwargs,
        )
