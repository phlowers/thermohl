# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Any

from thermohl.power import ieee
from thermohl import floatArrayLike, intArrayLike


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def __init__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        precomputed_solar_radiation: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Init with args.

        See ieee.SolarHeating; it is exactly the same with altitude and
        turbidity set to zero. If more than one input are numpy arrays, they
        should have the same size.

        Args:
            latitude (float | numpy.ndarray): Latitude.
            altitude (float | numpy.ndarray): Altitude.
            cable_azimuth (float | numpy.ndarray): Azimuth.
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (solar, must be between 0 and 23).
            outer_diameter (float | numpy.ndarray): external diameter.
            solar_absorptivity (float | numpy.ndarray): Solar absorption coefficient.
            precomputed_solar_radiation (float | numpy.ndarray | None): Optional precomputed solar radiation term.

        """
        if "turbidity" in kwargs.keys():
            kwargs.pop("turbidity")
        super().__init__(
            latitude=latitude,
            altitude=altitude,
            cable_azimuth=cable_azimuth,
            turbidity=0.0,
            month=month,
            day=day,
            hour=hour,
            outer_diameter=outer_diameter,
            solar_absorptivity=solar_absorptivity,
            precomputed_solar_radiation=precomputed_solar_radiation,
            **kwargs,
        )
