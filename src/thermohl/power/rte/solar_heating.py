# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import logging
from typing import Any, Tuple
import numpy as np
from thermohl import (
    floatArrayLike,
    sun,
    datetimeArrayLike,
)
from thermohl.power import SolarHeatingBase

from thermohl.utils import bisect_v
from thermohl import errors as thermohl_errors


logger = logging.getLogger(__name__)


TOL = 1e-06


def diffuse_and_beam_radiations(
    datetime_utc: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    nebulosity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diffuse radiation and beam radiation.

    Args:
        datetime_utc(np.ndarray): Array of datetimes (more precisely np.datetime64). The year is indifferent.
        latitude(np.ndarray): Array of latitudes.
        longitude(np.ndarray): Array of longitudes.
        nebulosity(np.ndarray): Array of nebulosities (integer between 0 and 8).
    Returns:
        tuple(np.ndarray, np.ndarray): diffuse_radiation, beam_radiation in W/m².
    """
    if (nebulosity < 0).any() or (nebulosity > 8).any():
        raise ValueError(f"nebulosity must be between 0 and 8. Got {nebulosity}")

    solar_hour = sun.utc2solar_hour(datetime_utc, np.deg2rad(longitude))
    solar_altitude = sun.solar_altitude(np.deg2rad(latitude), datetime_utc, solar_hour)
    global_radiation = compute_global_radiation(solar_altitude, nebulosity)
    diffuse_radiation = compute_diffuse_radiation(global_radiation, nebulosity)
    beam_radiation = compute_beam_radiation(
        global_radiation, diffuse_radiation, solar_altitude
    )
    return diffuse_radiation, beam_radiation


def estimate_nebulosity(
    diffuse_plus_beam_radiation: np.ndarray,
    datetime_utc: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.array:
    """Estimate nebulosity from measured diffuse radiation + beam radiation.

    The results are rounded to the values which give the closest radiation sums.

    Raises RadiationIncompatibleWithParametersError if it's impossible to have
    this radiation level with given parameters (datetime_utc, latitude and longitude).

    Args:
        diffuse_plus_beam_solar_flow(np.ndarray): Array of diffuse radiation + beam radiation (in W/m²).
        datetime_utc(np.ndarray): Array of datetimes (more precisely np.datetime64). The year is indifferent.
        latitude(np.ndarray): Array of latitudes.
        longitude(np.ndarray): Array of longitudes.
    Returns:
        np.ndarray: Nebulosities (integers between 0 and 8, or nan if it can't be computed because of the night).
    """
    solar_hour = sun.utc2solar_hour(datetime_utc, np.deg2rad(longitude))
    solar_altitude = sun.solar_altitude(np.deg2rad(latitude), datetime_utc, solar_hour)
    return estimate_nebulosity_from_diffuse_and_beam_radiation(
        solar_altitude,
        diffuse_plus_beam_radiation,
    )


def compute_global_radiation(
    solar_altitude: floatArrayLike, nebulosity: floatArrayLike
) -> floatArrayLike:
    res = (910 * np.sin(solar_altitude) - 30) * (
        1 - 0.75 * np.power(nebulosity / 8, 3.4)
    )
    res = np.maximum(0, res)
    # radiation is zero during the night
    return np.where(np.sin(solar_altitude) < TOL, 0, res)


def compute_diffuse_radiation(
    global_radiation: floatArrayLike, nebulosity: floatArrayLike
) -> floatArrayLike:
    return global_radiation * (0.3 + 0.7 * (nebulosity / 8) ** 2)


def compute_beam_radiation(
    global_radiation, diffuse_radiation, solar_altitude
) -> floatArrayLike:
    # if sin(solar altitude) <= 0 (dawn, twilight or night) the beam radiation is zero
    return np.where(
        np.sin(solar_altitude) <= 0,
        0,
        (global_radiation - diffuse_radiation) / np.sin(solar_altitude),
    )


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

    diffuse_radiation = compute_diffuse_radiation(global_radiation, nebulosity)
    beam_radiation = compute_beam_radiation(
        global_radiation, diffuse_radiation, solar_altitude
    )
    solar_irradiance = beam_radiation * (
        np.sin(incidence) + np.pi / 2 * albedo * np.sin(solar_altitude)
    ) + diffuse_radiation * np.pi / 2 * (1 + albedo)

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
    If solar altitude shows it is the night (or nearly the night, with solar altitude = 0), the global radiation is zero.

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
    final_global_radiation = compute_global_radiation(solar_altitude, final_nebulosity)

    return final_nebulosity, final_global_radiation


def estimate_nebulosity_from_diffuse_and_beam_radiation(
    solar_altitude: floatArrayLike, radiation_sum: floatArrayLike
) -> float:
    """Estimate nebulosity based on diffuse radiation + beam radiation, and solar altitude.

    For solar_altitude values corresponding to the night, the result is nan.
    Else, if no nebulosity could yield the given radiation (e.g. given radiation
    is too high for given solar altitude), it raises a RadiationIncompatibleWithParametersError.

    Args:
        solar_altitude(float): solar altitude in radians.
        radiation_sum(float): diffuse radiation + beam radiation.
    Returns:
        integer or np.nan: nebulosity (integer between 0 and 8).
    """

    # Since f is strictly monotonous (increasing) we can use dichotomy
    # algorithm to find x which minimizes |f|.
    def f(x):
        global_radiation = compute_global_radiation(solar_altitude, x)
        diffuse_radiation = compute_diffuse_radiation(global_radiation, x)
        beam_radiation = compute_beam_radiation(
            global_radiation, diffuse_radiation, solar_altitude
        )
        return radiation_sum - diffuse_radiation - beam_radiation

    lower_bound = 0
    upper_bound = 8

    if hasattr(radiation_sum, "shape") and radiation_sum.shape:
        output_shape = radiation_sum.shape
    else:
        output_shape = (1,)

    try:
        # Very few iterations are needed because we want an integer approximate answer
        nebulosity, _ = bisect_v(
            f, lower_bound, upper_bound, output_shape, max_iterations=4
        )
    except ValueError:
        raise thermohl_errors.RadiationIncompatibleWithParametersError()

    rounded_down = np.floor(nebulosity)
    rounded_up = np.ceil(nebulosity)
    nebulosity = np.where(
        np.abs(f(rounded_down)) <= np.abs(f(rounded_up)),
        rounded_down,
        rounded_up,
    )
    # negative sin(solar_altitude) means this is the night
    # so can't compute nebulosity
    return np.where(np.sin(solar_altitude) <= TOL, np.nan, nebulosity)


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
