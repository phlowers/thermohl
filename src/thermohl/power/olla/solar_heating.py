# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

from thermohl.power import ieee
from thermohl import floatArrayLike, datetimeListLike


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def __init__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        solar_irradiance: floatArrayLike,
        **kwargs: Any,
    ):
        """Init with args.
        See ieee.SolarHeating, it is exactly the same with altitude and turbidity set to zero.
        If more than one input are numpy arrays, they should have the same size.

        :param latitude: Latitude in degrees.
        :param altitude: Altitude.
        :param cable_azimuth: Azimuth of the conductor in degrees.
        :param datetime_utc: Datetime in UTC.
        :param outer_diameter: external diameter of the conductor.
        :param solar_absorptivity: Solar absorption coefficient of the conductor.
        :param solar_irradiance: Optional precomputed solar radiation term.
        """
        if "turbidity" in kwargs.keys():
            kwargs.pop("turbidity")
        super().__init__(
            latitude=latitude,
            altitude=altitude,
            cable_azimuth=cable_azimuth,
            turbidity=0.0,
            datetime_utc=datetime_utc,
            outer_diameter=outer_diameter,
            solar_absorptivity=solar_absorptivity,
            solar_irradiance=solar_irradiance,
            **kwargs,
        )
