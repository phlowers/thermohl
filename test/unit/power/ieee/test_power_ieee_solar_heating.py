# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power.ieee import SolarHeating


def test_solar_heating_init_scalar():
    latitude = 45.0
    altitude = 1000.0
    azimuth = 180.0
    turbidity = 0.5
    month = 6
    day = 21
    hour = 12.0
    outer_diameter = 0.01
    solar_absorptivity = 0.9
    srad = 800.0

    solar_heating = SolarHeating(
        latitude,
        altitude,
        azimuth,
        turbidity,
        month,
        day,
        hour,
        outer_diameter,
        solar_absorptivity,
        srad,
    )

    assert solar_heating.solar_absorptivity == solar_absorptivity
    assert np.isclose(solar_heating.solar_irradiance, srad)
    assert np.isclose(solar_heating.outer_diameter, outer_diameter)


def test_solar_heating_init_array():
    latitude = np.array([45.0, 50.0])
    altitude = np.array([1000.0, 2000.0])
    azimuth = np.array([180.0, 190.0])
    turbidity = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    outer_diameter = np.array([0.01, 0.02])
    solar_absorptivity = np.array([0.9, 0.8])
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(
        latitude,
        altitude,
        azimuth,
        turbidity,
        month,
        day,
        hour,
        outer_diameter,
        solar_absorptivity,
        srad,
    )

    assert np.allclose(solar_heating.solar_absorptivity, solar_absorptivity)
    assert np.allclose(solar_heating.solar_irradiance, srad)
    assert np.allclose(solar_heating.outer_diameter, outer_diameter)


def test_solar_heating_init_mixed():
    latitude = 45.0
    altitude = 1000.0
    azimuth = 180.0
    turbidity = 0.5
    month = 6
    day = 21
    hour = 12.0
    outer_diameter = 0.01
    solar_absorptivity = 0.9
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(
        latitude,
        altitude,
        azimuth,
        turbidity,
        month,
        day,
        hour,
        outer_diameter,
        solar_absorptivity,
        srad,
    )

    assert solar_heating.solar_absorptivity == solar_absorptivity
    assert np.allclose(solar_heating.solar_irradiance, srad)
    assert np.isclose(solar_heating.outer_diameter, outer_diameter)


def test_solar_heating_init_no_srad():
    latitude = 45.0
    altitude = 1000.0
    azimuth = 180.0
    turbidity = 0.5
    month = 6
    day = 21
    hour = 12.0
    outer_diameter = 0.01
    solar_absorptivity = 0.9

    solar_heating = SolarHeating(
        latitude,
        altitude,
        azimuth,
        turbidity,
        month,
        day,
        hour,
        outer_diameter,
        solar_absorptivity,
    )

    assert solar_heating.solar_absorptivity == solar_absorptivity
    assert solar_heating.solar_irradiance is not None
    assert np.isclose(solar_heating.outer_diameter, outer_diameter)
