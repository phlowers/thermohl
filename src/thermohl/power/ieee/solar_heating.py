# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any


from thermohl import floatArrayLike, datetimeListLike
from thermohl.power import _SRad, SolarHeatingBase


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        turbidity: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        measured_solar_irradiance: floatArrayLike,
        **kwargs: Any,
    ):
        """Init with args.
        If more than one input are numpy arrays, they should have the same size.

        :param latitude: Latitude in degrees.
        :param altitude: Altitude.
        :param cable_azimuth: Azimuth of the conductor in degrees.
        :param turbidity: Air pollution from 0 (clean) to 1 (polluted).
        :param datetime_utc: Datetime in UTC.
        :param outer_diameter: external diameter of the conductor.
        :param solar_absorptivity: Solar absorption coefficient of the conductor.
        :param measured_solar_irradiance: Optional precomputed solar radiation term.
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
            cable_azimuth,
            turbidity,
            datetime_utc,
            outer_diameter,
            solar_absorptivity,
            est,
            measured_solar_irradiance,
            **kwargs,
        )
