# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Any


from thermohl import floatArrayLike, intArrayLike
from thermohl.power import _SRad, SolarHeatingBase


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        turbidity: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        precomputed_solar_radiation: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            latitude (float | numpy.ndarray): Latitude.
            altitude (float | numpy.ndarray): Altitude.
            azimuth (float | numpy.ndarray): Azimuth.
            turbidity (float | numpy.ndarray): Air pollution from 0 (clean) to 1 (polluted).
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (solar, must be between 0 and 23).
            outer_diameter (float | numpy.ndarray): external diameter.
            solar_absorptivity (float | numpy.ndarray): Solar absorption coefficient.
            precomputed_solar_radiation (float | numpy.ndarray | None): Optional precomputed solar radiation term.
        """
        est = _SRad(
            [
                -4.22391e01,
                +6.38044e01,
                -1.9220e00,
                +3.46921e-02,
                -3.61118e-04,
                +1.94318e-06,
                -4.07608e-09,
            ],
            [
                +5.31821e01,
                +1.4211e01,
                +6.6138e-01,
                -3.1658e-02,
                +5.4654e-04,
                -4.3446e-06,
                +1.3236e-08,
            ],
        )
        super().__init__(
            latitude,
            altitude,
            azimuth,
            turbidity,
            month,
            day,
            hour,
            outer_diameter,
            solar_absorptivity,
            est,
            precomputed_solar_radiation,
            **kwargs,
        )
