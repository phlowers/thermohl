# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from math import pi
from typing import Any, Tuple, Iterable
import numpy as np
from thermohl import (
    floatArrayLike,
    sun,
    datetimeListLike,
)
from thermohl.power import SolarHeatingBase


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

    :param solar_altitude_rad: solar altitude in radians.
    :return: Solar radiation value. Negative values are set to zero.
    """

    def compute_diffuse_radiation() -> floatArrayLike:
        return global_radiation * (0.3 + 0.7 * (nebulosity / 8) ** 2)

    def compute_beam_radiation() -> floatArrayLike:
        # si l'altitude solaire est nulle, on fixe la radiation solaire à 0.
        with np.errstate(divide="ignore", invalid="ignore"):
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

    def compute_nebulosity(provided_global_radiation, solar_altitude):
        intermediate = np.minimum(
            1, provided_global_radiation[~mask] / (910 * np.sin(solar_altitude) - 30)
        )
        nebulosity = 8 * (4 / 3 * (1 - intermediate)) ** (1 / 3.4)
        return np.minimum(8, nebulosity)

    def compute_global_radiation(provided_nebulosity, solar_altitude):
        intermediate = 1 - 3 / 4 * (provided_nebulosity / 8) ** 3.4
        global_radiation = (910 * np.sin(solar_altitude) - 30) * intermediate
        return np.maximum(0, global_radiation)

    mask = np.isnan(provided_global_radiation)

    nebulosity = np.empty_like(provided_global_radiation)
    nebulosity[mask] = provided_nebulosity[mask]
    if np.any(~mask):
        nebulosity[~mask] = compute_nebulosity(
            provided_global_radiation[~mask], solar_altitude[~mask]
        )

    global_radiation = np.empty_like(provided_global_radiation)
    global_radiation[~mask] = provided_global_radiation[~mask]
    if np.any(mask):
        global_radiation[mask] = compute_global_radiation(
            provided_nebulosity[mask], solar_altitude[mask]
        )

    return nebulosity, global_radiation


class SolarHeating(SolarHeatingBase):

    def __init__(
        self,
        latitude: floatArrayLike,
        longitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        datetime_utc: datetimeListLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        albedo: floatArrayLike,
        nebulosity: floatArrayLike,
        measured_solar_irradiance: floatArrayLike,
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
        date = (
            [d.date() for d in datetime_utc]
            if isinstance(datetime_utc, Iterable)
            else datetime_utc.date()
        )
        solar_hour = sun.utc2solar_hour(datetime_utc, np.deg2rad(longitude))
        solar_altitude = sun.solar_altitude(np.deg2rad(latitude), date, solar_hour)
        nebulosity, global_radiation = compute_data_from_provided(
            measured_solar_irradiance, nebulosity, solar_altitude
        )
        solar_azimuth_rad = sun.solar_azimuth(np.deg2rad(latitude), date, solar_hour)
        incidence = np.arccos(
            np.cos(solar_altitude)
            * np.cos(solar_azimuth_rad - np.deg2rad(cable_azimuth))
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
