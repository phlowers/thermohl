# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl.power.cigre import Air


def test_volumic_mass_scalar():
    air_temperature = 25.0
    altitude = 1000.0
    expected = 1.2925 * np.exp(-1.16e-04 * altitude)

    result = Air.volumic_mass(air_temperature, altitude)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    expected = 1.2925 * np.exp(-1.16e-04 * altitude)

    result = Air.volumic_mass(air_temperature, altitude)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_default_altitude():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = 1.2925 * np.ones_like(air_temperature)

    result = Air.volumic_mass(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_mismatched_array_sizes():
    air_temperature = np.array([15.0, 20.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.volumic_mass(air_temperature, altitude)


def test_relative_density_scalar():
    air_temperature = 25.0
    altitude = 1000.0
    expected = np.exp(-1.16e-04 * altitude)

    result = Air.relative_density(air_temperature, altitude)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    expected = np.exp(-1.16e-04 * altitude)

    result = Air.relative_density(air_temperature, altitude)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_default_altitude():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = np.ones_like(air_temperature)

    result = Air.relative_density(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_mismatched_array_sizes():
    air_temperature = np.array([15.0, 20.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.relative_density(air_temperature, altitude)


def test_kinematic_viscosity_scalar():
    air_temperature = 25.0
    expected = 1.32e-05 + 9.5e-08 * air_temperature

    result = Air.kinematic_viscosity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = 1.32e-05 + 9.5e-08 * air_temperature

    result = Air.kinematic_viscosity(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_negative_temperature():
    air_temperature = -10.0
    expected = 1.32e-05 + 9.5e-08 * air_temperature

    result = Air.kinematic_viscosity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_zero_temperature():
    air_temperature = 0.0
    expected = 1.32e-05

    result = Air.kinematic_viscosity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_scalar():
    air_temperature = 25.0
    altitude = 1000.0
    expected = Air.kinematic_viscosity(air_temperature) * Air.volumic_mass(
        air_temperature, altitude
    )

    result = Air.dynamic_viscosity(air_temperature, altitude)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    expected = Air.kinematic_viscosity(air_temperature) * Air.volumic_mass(
        air_temperature, altitude
    )

    result = Air.dynamic_viscosity(air_temperature, altitude)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_default_altitude():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = Air.kinematic_viscosity(air_temperature) * Air.volumic_mass(
        air_temperature
    )

    result = Air.dynamic_viscosity(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_mismatched_array_sizes():
    air_temperature = np.array([15.0, 20.0])
    altitude = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.dynamic_viscosity(air_temperature, altitude)


def test_thermal_conductivity_scalar():
    air_temperature = 25.0
    expected = 2.42e-02 + 7.2e-05 * air_temperature

    result = Air.thermal_conductivity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = 2.42e-02 + 7.2e-05 * air_temperature

    result = Air.thermal_conductivity(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_negative_temperature():
    air_temperature = -10.0
    expected = 2.42e-02 + 7.2e-05 * air_temperature

    result = Air.thermal_conductivity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_zero_temperature():
    air_temperature = 0.0
    expected = 2.42e-02

    result = Air.thermal_conductivity(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_scalar():
    air_temperature = 25.0
    expected = 0.715 - 2.5e-04 * air_temperature

    result = Air.prandtl(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_array():
    air_temperature = np.array([15.0, 20.0, 25.0])
    expected = 0.715 - 2.5e-04 * air_temperature

    result = Air.prandtl(air_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_negative_temperature():
    air_temperature = -10.0
    expected = 0.715 - 2.5e-04 * air_temperature

    result = Air.prandtl(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_zero_temperature():
    air_temperature = 0.0
    expected = 0.715

    result = Air.prandtl(air_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
