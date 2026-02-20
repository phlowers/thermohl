# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.cigre import ConvectiveCooling
from thermohl.power.cigre import Air


conv_cool_instances = [
    (
        ConvectiveCooling(
            altitude=np.array([100.0]),
            azimuth=np.array([45.0]),
            ambient_temperature=np.array([25.0]),
            wind_speed=np.array([5.0]),
            wind_angle=np.array([30.0]),
            outer_diameter=np.array([0.01]),
            roughness_ratio=np.array([0.02]),
        ),
        np.ndarray,
    ),
    (
        ConvectiveCooling(
            altitude=100.0,
            azimuth=45.0,
            ambient_temperature=25.0,
            wind_speed=5.0,
            wind_angle=30.0,
            outer_diameter=0.01,
            roughness_ratio=0.02,
        ),
        float,
    ),
]


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_forced_float_value(convective_cooling, expected_type):
    Tf = 30.0
    nu = 1.5e-5
    expected_result = 17.3425

    result = convective_cooling._nu_forced(Tf, nu)

    np.testing.assert_allclose(result, expected_result, rtol=1e-5)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_forced_array_single_value(convective_cooling, expected_type):
    Tf = np.array([30.0])
    nu = np.array([1.5e-5])
    expected_result = np.array([17.3425])

    result = convective_cooling._nu_forced(Tf, nu)

    np.testing.assert_allclose(result, expected_result, rtol=1e-5)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_forced_array_values(convective_cooling, expected_type):
    Tf = np.array([30.0, 35.0])
    nu = np.array([1.5e-5, 1.6e-5])
    expected_result = np.array([17.3425, 16.6482])

    result = convective_cooling._nu_forced(Tf, nu)

    np.testing.assert_allclose(result, expected_result, rtol=1e-5)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_forced_boundary_conditions(convective_cooling, expected_type):
    Tf = np.array([30.0])
    nu = np.array([1.5e-5])
    convective_cooling.roughness_ratio = np.array([0.05])
    convective_cooling.wind_speed = np.array([100.0])
    expected_result = np.array([115.5214])

    result = convective_cooling._nu_forced(Tf, nu)

    np.testing.assert_allclose(result, expected_result, rtol=1e-5)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_natural_single_value(convective_cooling, expected_type):
    air_temperature = 50.0
    Td = 25.0
    nu = Air.kinematic_viscosity(air_temperature)

    result = convective_cooling._nu_natural(air_temperature, Td, nu)

    assert isinstance(result, expected_type)
    np.testing.assert_allclose(result, [3.42404], rtol=1e-4)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_natural_array_values(convective_cooling, expected_type):
    air_temperature = np.array([50.0, 60.0, 70.0])
    Td = np.array([25.0, 35.0, 45.0])
    nu = Air.kinematic_viscosity(air_temperature)

    result = convective_cooling._nu_natural(air_temperature, Td, nu)

    assert isinstance(result, np.ndarray)
    assert result.shape == air_temperature.shape
    np.testing.assert_allclose(result, [3.42404, 3.55476, 3.63592], rtol=1e-4)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_natural_edge_case(convective_cooling, expected_type):
    air_temperature = 0.0
    Td = 0.0
    nu = Air.kinematic_viscosity(air_temperature)

    result = convective_cooling._nu_natural(air_temperature, Td, nu)

    assert isinstance(result, expected_type)
    assert result == 0.0


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_nu_natural_high_values(convective_cooling, expected_type):
    air_temperature = 1000.0
    Td = 500.0
    nu = Air.kinematic_viscosity(air_temperature)

    result = convective_cooling._nu_natural(air_temperature, Td, nu)

    assert isinstance(result, expected_type)
    np.testing.assert_allclose(result, [2.18853], rtol=1e-4)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_value_single_value(convective_cooling, expected_type):
    conductor_temperature = 75.0

    result = convective_cooling.value(conductor_temperature)

    assert isinstance(result, np.ndarray)
    assert result > 0


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_value_array_values(convective_cooling, expected_type):
    conductor_temperature = np.array([75.0, 85.0, 95.0])

    result = convective_cooling.value(conductor_temperature)

    assert isinstance(result, np.ndarray)
    assert result.shape == conductor_temperature.shape
    assert np.all(result > 0)


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_value_edge_case(convective_cooling, expected_type):
    conductor_temperature = 25.0  # Same as ambient temperature

    result = convective_cooling.value(conductor_temperature)

    assert isinstance(result, np.ndarray)
    assert result == 0.0


@pytest.mark.parametrize(
    "convective_cooling, expected_type",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_value_high_values(convective_cooling, expected_type):
    conductor_temperature = 1000.0

    result = convective_cooling.value(conductor_temperature)

    assert isinstance(result, np.ndarray)
    assert result > 0
