# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power.ieee import SolarHeating


def test_solar_heating_init_scalar():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = 800.0

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert solar_heating.absorption_coefficient == alpha
    assert np.isclose(solar_heating.solar_radiation, srad)
    assert np.isclose(solar_heating.conductor_diameter_m, D)


def test_solar_heating_init_array():
    lat = np.array([45.0, 50.0])
    alt = np.array([1000.0, 2000.0])
    azm = np.array([180.0, 190.0])
    tb = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    D = np.array([0.01, 0.02])
    alpha = np.array([0.9, 0.8])
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert np.allclose(solar_heating.absorption_coefficient, alpha)
    assert np.allclose(solar_heating.solar_radiation, srad)
    assert np.allclose(solar_heating.conductor_diameter_m, D)


def test_solar_heating_init_mixed():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9
    srad = np.array([800.0, 900.0])

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha, srad)

    assert solar_heating.absorption_coefficient == alpha
    assert np.allclose(solar_heating.solar_radiation, srad)
    assert np.isclose(solar_heating.conductor_diameter_m, D)


def test_solar_heating_init_no_srad():
    lat = 45.0
    alt = 1000.0
    azm = 180.0
    tb = 0.5
    month = 6
    day = 21
    hour = 12.0
    D = 0.01
    alpha = 0.9

    solar_heating = SolarHeating(lat, alt, azm, tb, month, day, hour, D, alpha)

    assert solar_heating.absorption_coefficient == alpha
    assert solar_heating.solar_radiation is not None
    assert np.isclose(solar_heating.conductor_diameter_m, D)
