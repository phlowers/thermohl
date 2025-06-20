# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power.olla import Air


def test_volumic_mass_scalar():
    temp_celsius = 25.0
    altitude_meters = 0.0
    expected = 1.184

    result = Air.volumic_mass(temp_celsius, altitude_meters)

    assert np.isclose(result, expected, rtol=1e-2)


def test_volumic_mass_array():
    temp_celsius = np.array([0.0, 25.0, 100.0])
    altitude_meters = np.array([0.0, 1000.0, 2000.0])
    expected = np.array([1.292, 1.055, 0.787])

    result = Air.volumic_mass(temp_celsius, altitude_meters)

    assert np.allclose(result, expected, rtol=1e-2)


def test_volumic_mass_altitude_scalar():
    temp_celsius = 25.0
    altitude_meters = 1000.0
    expected = 1.055

    result = Air.volumic_mass(temp_celsius, altitude_meters)

    assert np.isclose(result, expected, rtol=1e-2)


def test_volumic_mass_altitude_array():
    temp_celsius = np.array([0.0, 25.0, 100.0])
    altitude_meters = 1000.0
    expected = np.array([1.139, 1.055, 0.862])

    result = Air.volumic_mass(temp_celsius, altitude_meters)

    assert np.allclose(result, expected, rtol=1e-2)


def test_dynamic_viscosity_scalar():
    temp_celsius = 25.0
    expected = 1.849e-05

    result = Air.dynamic_viscosity(temp_celsius)

    assert np.isclose(result, expected, rtol=1e-2)


def test_dynamic_viscosity_array():
    temp_celsius = np.array([0.0, 25.0, 100.0])
    expected = np.array([1.723e-05, 1.839e-05, 2.168e-05])

    result = Air.dynamic_viscosity(temp_celsius)

    assert np.allclose(result, expected, rtol=1e-2)


def test_kinematic_viscosity_scalar():
    temp_celsius = 25.0
    altitude_meters = 0.0
    expected = 1.561e-05

    result = Air.kinematic_viscosity(temp_celsius, altitude_meters)

    assert np.isclose(result, expected, rtol=1e-2)


def test_kinematic_viscosity_array():
    temp_celsius = np.array([0.0, 25.0, 100.0])
    altitude_meters = np.array([0.0, 1000.0, 2000.0])
    expected = np.array([1.333e-05, 1.742e-05, 2.754e-05])

    result = Air.kinematic_viscosity(temp_celsius, altitude_meters)

    assert np.allclose(result, expected, rtol=1e-2)


def test_thermal_conductivity_scalar():
    temp_celsius = 25.0
    expected = 0.02587

    result = Air.thermal_conductivity(temp_celsius)

    assert np.isclose(result, expected, rtol=1e-2)


def test_thermal_conductivity_array():
    temp_celsius = np.array([0.0, 25.0, 100.0])
    expected = np.array([0.024, 0.026, 0.0316])

    result = Air.thermal_conductivity(temp_celsius)

    assert np.allclose(result, expected, rtol=1e-2)
