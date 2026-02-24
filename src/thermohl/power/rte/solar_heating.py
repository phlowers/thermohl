# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from math import pi
from typing import Optional, Any, Tuple

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun
from thermohl.power import SolarHeatingBase, _SRad


def compute_solar_irradiance(
    global_radiation: floatArrayLike,
    solar_altitude: floatArrayLike,
    incidence: floatArrayLike,
    nebulosity: floatArrayLike,
    albedo: floatArrayLike,
) -> floatArrayLike:
    """Compute solar radiation.
    Difference with IEEE version are neither turbidity or altitude influence.
    Args:
        solar_altitude (float | numpy.ndarray): solar altitude in radians.
    Returns:
        float | numpy.ndarray: Solar radiation value. Negative values are set to zero.
    """

    def compute_diffuse_radiation() -> floatArrayLike:
        return global_radiation * (0.3 + 0.7 * (nebulosity / 8) ** 2)

    def compute_beam_radiation() -> floatArrayLike:
        return (global_radiation - diffuse_radiation) / np.sin(solar_altitude)

    diffuse_radiation = compute_diffuse_radiation()
    beam_radiation = compute_beam_radiation()
    solar_irradiance = beam_radiation * (
        np.sin(incidence) + pi / 2 * albedo * np.sin(solar_altitude)
    ) + diffuse_radiation * pi / 2 * (1 + albedo)

    return np.where(solar_altitude > 0.0, solar_irradiance, 0.0)


def compute_data_from_provided(
    provided_global_radiation: floatArrayLike,
    provided_nebulosity: floatArrayLike,
    solar_altitude: floatArrayLike,
) -> Tuple[floatArrayLike, floatArrayLike]:
    """
    Returns a value of nebulosity and a value of global_radiation.
    If the global radiation is provided, the nebulosity is computed from it.
    Otherwise, the global radiation is computed from the provided nebulosity (default value of 0).
    """

    def compute_nebulosity():
        intermediate = np.min(
            1, provided_global_radiation / (910 * np.sin(solar_altitude) - 30)
        )
        nebulosity = 8 * (4 / 3 * (1 - intermediate)) ** (1 / 3.4)
        return np.round(np.min(8, nebulosity))

    def compute_global_radiation():
        intermediate = 1 - 3 / 4 * (provided_nebulosity / 8) ** 3.4
        global_radiation = (910 * np.sin(solar_altitude) - 30) * intermediate
        return np.max(0, global_radiation)

    nebulosity = np.where(
        np.isnan(provided_global_radiation),
        provided_nebulosity,
        compute_nebulosity(),
    )

    global_radiation = np.where(
        np.isnan(provided_global_radiation),
        compute_global_radiation(),
        provided_global_radiation,
    )

    return nebulosity, global_radiation


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
        albedo: floatArrayLike,
        nebulosity: floatArrayLike,
        measured_solar_irradiance: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Build with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            latitude (float | numpy.ndarray): Latitude.
            longitude (float | numpy.ndarray): Longitude (must be between -180 and +180 degrees).
            azimuth (float | numpy.ndarray): Azimuth.
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (must be between 0 and 24 excluded).
            outer_diameter (float | numpy.ndarray): external diameter.
            solar_absorptivity (numpy.ndarray): Solar absorption coefficient.
            measured_solar_irradiance (float | numpy.ndarray | None): Optional measured solar irradiance (W/m2).
        """
        solar_hour = sun.utc2solar_hour(hour, day, month, np.deg2rad(longitude))
        solar_altitude = sun.solar_altitude(
            np.deg2rad(latitude), month, day, solar_hour
        )
        nebulosity, global_radiation = compute_data_from_provided(
            measured_solar_irradiance, nebulosity, solar_altitude
        )
        solar_azimuth_rad = sun.solar_azimuth(np.deg2rad(latitude), month, day, hour)
        incidence = np.cos(solar_altitude) * np.cos(
            solar_azimuth_rad - np.deg2rad(azimuth)
        )

        self.solar_absorptivity = solar_absorptivity
        self.outer_diameter = outer_diameter
        self.solar_irradiance = compute_solar_irradiance(
            global_radiation,
            solar_altitude,
            incidence,
            nebulosity,
            albedo,
        )
