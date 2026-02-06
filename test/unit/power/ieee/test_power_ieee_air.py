# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power.ieee import Air


def test_volumic_mass_scalar():
    air_temperature = 25.0
    altitude = 1000.0
    expected = (1.293 - 1.525e-04 * altitude + 6.379e-09 * altitude**2) / (
        1.0 + 0.00367 * air_temperature
    )

    result = Air.volumic_mass(air_temperature, altitude)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array():
    air_temperature = np.array([25.0, 30.0])
    altitude = np.array([1000.0, 2000.0])
    expected = (1.293 - 1.525e-04 * altitude + 6.379e-09 * altitude**2) / (
        1.0 + 0.00367 * air_temperature
    )

    result = Air.volumic_mass(air_temperature, altitude)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_default_altitude():
    air_temperature = 25.0
    expected = (1.293 - 1.525e-04 * 0.0 + 6.379e-09 * 0.0**2) / (
        1.0 + 0.00367 * air_temperature
    )

    result = Air.volumic_mass(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array_default_altitude():
    air_temperature = np.array([25.0, 30.0])
    expected = (1.293 - 1.525e-04 * 0.0 + 6.379e-09 * 0.0**2) / (
        1.0 + 0.00367 * air_temperature
    )

    result = Air.volumic_mass(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_scalar():
    air_temperature = 25.0
    expected = (1.458e-06 * (air_temperature + 273.0) ** 1.5) / (
        air_temperature + 383.4
    )

    result = Air.dynamic_viscosity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_array():
    air_temperature = np.array([25.0, 30.0])
    expected = (1.458e-06 * (air_temperature + 273.0) ** 1.5) / (
        air_temperature + 383.4
    )

    result = Air.dynamic_viscosity(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_scalar():
    air_temperature = 25.0
    expected = 2.424e-02 + 7.477e-05 * air_temperature - 4.407e-09 * air_temperature**2

    result = Air.thermal_conductivity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_array():
    air_temperature = np.array([25.0, 30.0])
    expected = 2.424e-02 + 7.477e-05 * air_temperature - 4.407e-09 * air_temperature**2

    result = Air.thermal_conductivity(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
