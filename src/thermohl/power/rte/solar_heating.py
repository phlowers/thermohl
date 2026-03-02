# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Optional, Any

import numpy as np

from thermohl import (
    floatArrayLike,
    sun,
    datetimeListLike,
)
from thermohl.power import SolarHeatingBase, _SRad

CLEAN_AIR_COEFFICIENTS = [
    -42.0,
    +63.8,
    -1.922,
    0.03469,
    -3.61e-04,
    +1.943e-06,
    -4.08e-09,
]
POLLUTED_AIR_COEFFICIENTS = [0, 0, 0, 0, 0, 0, 0]

solar_radiation = _SRad(clean=CLEAN_AIR_COEFFICIENTS, indus=POLLUTED_AIR_COEFFICIENTS)


def solar_irradiance(
    solar_altitude_rad: floatArrayLike,
) -> floatArrayLike:
    """Compute solar radiation.
    Difference with IEEE version are neither turbidity or altitude influence.

    :param solar_altitude_rad: solar altitude in radians.
    :return: Solar radiation value. Negative values are set to zero.
    """

    clearness_factor = solar_radiation.atmosphere_turbidity(
        np.rad2deg(solar_altitude_rad)
    )
    return np.where(solar_altitude_rad > 0.0, clearness_factor, 0.0)


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        latitude: floatArrayLike,
        longitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        measured_solar_irradiance: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        """Build with args.
        If more than one input are numpy arrays, they should have the same size.

        :param latitude: Latitude in degrees.
        :param longitude: Longitude in degrees (must be between -180 and +180 degrees).
        :param cable_azimuth: Azimuth of the conductor in degrees.
        :param datetime_utc: Datetime in UTC.
        :param outer_diameter: external diameter of the conductor.
        :param solar_absorptivity: Solar absorption coefficient of the conductor.
        :param measured_solar_irradiance: Optional measured solar irradiance (W/m2).
        """
        self.solar_absorptivity = solar_absorptivity
        solar_hour = sun.utc2solar_hour(datetime_utc, np.deg2rad(longitude))
        date = [d.date() for d in datetime_utc]
        solar_altitude_rad = sun.solar_altitude(np.deg2rad(latitude), date, solar_hour)
        if np.isnan(measured_solar_irradiance).all():
            measured_solar_irradiance = solar_irradiance(solar_altitude_rad)
        solar_azimuth_rad = sun.solar_azimuth(np.deg2rad(latitude), date, solar_hour)
        incidence_angle_rad = np.arccos(
            np.cos(solar_altitude_rad)
            * np.cos(solar_azimuth_rad - np.deg2rad(cable_azimuth))
        )
        irradiance = measured_solar_irradiance * np.sin(incidence_angle_rad)
        self.solar_irradiance = np.maximum(irradiance, 0.0)
        self.outer_diameter = outer_diameter
