# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.cigre import Air


def test_volumic_mass_scalar():
    Tc = 25.0
    alt = 1000.0
    expected = 1.2925 * np.exp(-1.16e-04 * alt)

    result = Air.volumic_mass(Tc, alt)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_array():
    Tc = np.array([15.0, 20.0, 25.0])
    alt = np.array([0.0, 500.0, 1000.0])
    expected = 1.2925 * np.exp(-1.16e-04 * alt)

    result = Air.volumic_mass(Tc, alt)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_default_altitude():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = 1.2925 * np.ones_like(Tc)

    result = Air.volumic_mass(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_volumic_mass_mismatched_array_sizes():
    Tc = np.array([15.0, 20.0])
    alt = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.volumic_mass(Tc, alt)


def test_relative_density_scalar():
    Tc = 25.0
    alt = 1000.0
    expected = np.exp(-1.16e-04 * alt)

    result = Air.relative_density(Tc, alt)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_array():
    Tc = np.array([15.0, 20.0, 25.0])
    alt = np.array([0.0, 500.0, 1000.0])
    expected = np.exp(-1.16e-04 * alt)

    result = Air.relative_density(Tc, alt)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_default_altitude():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = np.ones_like(Tc)

    result = Air.relative_density(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_relative_density_mismatched_array_sizes():
    Tc = np.array([15.0, 20.0])
    alt = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.relative_density(Tc, alt)


def test_kinematic_viscosity_scalar():
    Tc = 25.0
    expected = 1.32e-05 + 9.5e-08 * Tc

    result = Air.kinematic_viscosity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_array():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = 1.32e-05 + 9.5e-08 * Tc

    result = Air.kinematic_viscosity(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_negative_temperature():
    Tc = -10.0
    expected = 1.32e-05 + 9.5e-08 * Tc

    result = Air.kinematic_viscosity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_kinematic_viscosity_zero_temperature():
    Tc = 0.0
    expected = 1.32e-05

    result = Air.kinematic_viscosity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_scalar():
    Tc = 25.0
    alt = 1000.0
    expected = Air.kinematic_viscosity(Tc) * Air.volumic_mass(Tc, alt)

    result = Air.dynamic_viscosity(Tc, alt)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_array():
    Tc = np.array([15.0, 20.0, 25.0])
    alt = np.array([0.0, 500.0, 1000.0])
    expected = Air.kinematic_viscosity(Tc) * Air.volumic_mass(Tc, alt)

    result = Air.dynamic_viscosity(Tc, alt)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_default_altitude():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = Air.kinematic_viscosity(Tc) * Air.volumic_mass(Tc)

    result = Air.dynamic_viscosity(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_dynamic_viscosity_mismatched_array_sizes():
    Tc = np.array([15.0, 20.0])
    alt = np.array([0.0, 500.0, 1000.0])
    with pytest.raises(ValueError):
        Air.dynamic_viscosity(Tc, alt)


def test_thermal_conductivity_scalar():
    Tc = 25.0
    expected = 2.42e-02 + 7.2e-05 * Tc

    result = Air.thermal_conductivity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_array():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = 2.42e-02 + 7.2e-05 * Tc

    result = Air.thermal_conductivity(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_negative_temperature():
    Tc = -10.0
    expected = 2.42e-02 + 7.2e-05 * Tc

    result = Air.thermal_conductivity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_thermal_conductivity_zero_temperature():
    Tc = 0.0
    expected = 2.42e-02

    result = Air.thermal_conductivity(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_scalar():
    Tc = 25.0
    expected = 0.715 - 2.5e-04 * Tc

    result = Air.prandtl(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_array():
    Tc = np.array([15.0, 20.0, 25.0])
    expected = 0.715 - 2.5e-04 * Tc

    result = Air.prandtl(Tc)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_negative_temperature():
    Tc = -10.0
    expected = 0.715 - 2.5e-04 * Tc

    result = Air.prandtl(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_prandtl_zero_temperature():
    Tc = 0.0
    expected = 0.715

    result = Air.prandtl(Tc)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
