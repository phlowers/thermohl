# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from math import pi
import logging
from typing import Any, Tuple
import numpy as np
from thermohl import (
    floatArrayLike,
    sun,
    datetimeArrayLike,
)
from thermohl.power import SolarHeatingBase


logger = logging.getLogger(__name__)


def compute_solar_irradiance(
    global_radiation: floatArrayLike,
    solar_altitude: floatArrayLike,
    incidence: floatArrayLike,
    nebulosity: floatArrayLike,
    albedo: floatArrayLike,
) -> floatArrayLike:
    """Compute solar radiation.
    Difference with IEEE version are neither turbidity or altitude influence.

    :param global_radiation: global radiation.
    :param solar_altitude: solar altitude in radians.
    :param incidence: incidence angle in radians.
    :param nebulosity: nebulosity.
    :param albedo: albedo.
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
    If the nebulosity is provided, it is kept.
    Otherwise, if the global radiation is provided (ie not NaN), the nebulosity is computed from it.
    Otherwise, the nebulosity defaut value is 0.
    The returned global radiation is computed from the nebulosity, even if a global radiation is already provided.

    :param provided_global_radiation: provided global radiation (W/m2).
    :param provided_nebulosity: provided nebulosity (0 to 8).
    :param solar_altitude: solar altitude in radians.
    :return: (nebulosity, global_radiation).
    """

    # First, everything is computed (np.errstate ignore warnings on NaN values)
    with np.errstate(divide="ignore", invalid="ignore"):
        # nebulosity computation from global radiation
        inter_neb = np.minimum(
            1, provided_global_radiation / (910 * np.sin(solar_altitude) - 30)
        )
        computed_nebulosity = np.minimum(8, 8 * (4 / 3 * (1 - inter_neb)) ** (1 / 3.4))

    # Then, a filter is applied to keep useful values amongst the computations.
    final_nebulosity = np.where(
        ~np.isnan(provided_nebulosity),
        provided_nebulosity,
        np.where(~np.isnan(provided_global_radiation), computed_nebulosity, 0.0),
    )

    # Finally, the returned global radiation is computed from the previous nebulosity.
    with np.errstate(divide="ignore", invalid="ignore"):
        inter_rad = 1 - 3 / 4 * (final_nebulosity / 8) ** 3.4
        final_global_radiation = np.maximum(
            0, (910 * np.sin(solar_altitude) - 30) * inter_rad
        )

    return final_nebulosity, final_global_radiation


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        latitude: floatArrayLike,
        longitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        datetime_utc: datetimeArrayLike,
        outer_diameter: floatArrayLike,
        solar_absorptivity: floatArrayLike,
        albedo: floatArrayLike,
        nebulosity: floatArrayLike,
        measured_global_radiation: floatArrayLike,
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
        :param albedo: Ground albedo.
        :param nebulosity: Sky nebulosity (0 to 8).
        :param measured_global_radiation: Optional measured global radiation (W/m2) used to compute solar irradiance.
        """
        if (
            kwargs.get("solar_irradiance", None) is not None
            and not np.isnan(kwargs["solar_irradiance"]).all()
        ):
            logger.warning(
                "Got 'solar_irradiance' keyword argument in SolarHeating.__init__, which is not supported by Rte "
                "implementation. This will be ignored."
            )
            kwargs.pop("solar_irradiance")

        date = datetime_utc.astype("datetime64[D]")
        solar_hour = sun.utc2solar_hour(datetime_utc, np.deg2rad(longitude))
        solar_altitude = sun.solar_altitude(np.deg2rad(latitude), date, solar_hour)
        nebulosity, global_radiation = compute_data_from_provided(
            measured_global_radiation, nebulosity, solar_altitude
        )
        solar_azimuth_rad = sun.solar_azimuth(np.deg2rad(latitude), date, solar_hour)
        incidence = np.arccos(
            np.cos(solar_altitude)
            * np.cos(solar_azimuth_rad - np.deg2rad(cable_azimuth))
        )

        self.solar_absorptivity = solar_absorptivity
        self.outer_diameter = outer_diameter
        self.global_radiation = global_radiation
        self.solar_irradiance = compute_solar_irradiance(
            global_radiation,
            solar_altitude,
            incidence,
            nebulosity,
            albedo,
        )
