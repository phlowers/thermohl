# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun
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
    Args:
        solar_altitude_rad (float | numpy.ndarray): solar altitude in radians.
    Returns:
        float | numpy.ndarray: Solar radiation value. Negative values are set to zero.
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
        azimuth: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        measured_solar_irradiance: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Build with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            latitude (float | numpy.ndarray): Latitude.
            azimuth (float | numpy.ndarray): Azimuth.
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (must be between 0 and 24 excluded).
            outer_diameter (float | numpy.ndarray): external diameter.
            solar_absorptivity (numpy.ndarray): Solar absorption coefficient.
            measured_solar_irradiance (float | numpy.ndarray | None): Optional measured solar irradiance (W/m2).
        """
        self.solar_absorptivity = solar_absorptivity
        solar_hour = sun.utc2solar_hour(hour, day, month, np.deg2rad(longitude))
        solar_altitude_rad = sun.solar_altitude(
            np.deg2rad(latitude), month, day, solar_hour
        )
        if np.isnan(measured_solar_irradiance).all():
            measured_solar_irradiance = solar_irradiance(solar_altitude_rad)
        solar_azimuth_rad = sun.solar_azimuth(np.deg2rad(latitude), month, day, hour)
        incidence_angle_rad = np.arccos(
            np.cos(solar_altitude_rad) * np.cos(solar_azimuth_rad - np.deg2rad(azimuth))
        )
        irradiance = measured_solar_irradiance * np.sin(incidence_angle_rad)
        self.solar_irradiance = np.maximum(irradiance, 0.0)
        self.outer_diameter = outer_diameter
