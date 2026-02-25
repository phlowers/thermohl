# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl import solver
from thermohl.power.ieee import ConvectiveCooling


def set_default_values_scalar():
    dic = solver.default_values()
    dic["wind_speed"] = 0.61
    dic["wind_azimuth"] = 0.0
    dic["emissivity"] = 0.8
    dic["solar_absorptivity"] = 0.8
    dic["ambient_temperature"] = 40.0
    dic["temp_high"] = 75.0
    dic["temp_low"] = 25.0
    dic["linear_resistance_temp_high"] = 8.688e-05
    dic["linear_resistance_temp_low"] = 7.283e-05
    dic["cable_azimuth"] = 90.0
    dic["latitude"] = 30.0
    dic["turbidity"] = 0.0
    dic["altitude"] = 0.0
    dic["outer_diameter"] = 28.14 * 1.0e-03
    dic["core_diameter"] = 10.4 * 1.0e-03
    dic["month"] = 6
    dic["day"] = 10
    dic["hour"] = 11.0
    return dic


def set_default_values_array():
    dic = solver.default_values()
    dic["wind_speed"] = np.array([0.61, 0.83])
    dic["wind_azimuth"] = np.array([0.0, 42.1])
    dic["emissivity"] = np.array([0.8, 0.9])
    dic["solar_absorptivity"] = np.array([0.8, 0.9])
    dic["ambient_temperature"] = np.array([40.0, 32])
    dic["temp_high"] = np.array([75.0, 70.0])
    dic["temp_low"] = np.array([25.0, 20])
    dic["linear_resistance_temp_high"] = np.array([8.688e-05, 8.688e-05])
    dic["linear_resistance_temp_low"] = np.array([7.283e-05, 7.283e-05])
    dic["cable_azimuth"] = np.array([90.0, 90.0])
    dic["latitude"] = np.array([30.0, 30.0])
    dic["turbidity"] = np.array([0.0, 0.0])
    dic["altitude"] = np.array([0.0, 0.0])
    dic["outer_diameter"] = np.array([28.14 * 1.0e-03, 28.14 * 1.0e-03])
    dic["core_diameter"] = np.array([10.4 * 1.0e-03, 10.4 * 1.0e-03])
    dic["month"] = np.array([6, 3])
    dic["day"] = np.array([10, 13])
    dic["hour"] = np.array([11.0, 8.0])
    return dic


convective_cooling_instances = [
    (ConvectiveCooling(**set_default_values_array()), np.array([80.9093, 82.3549])),
    (
        ConvectiveCooling(
            **set_default_values_scalar(),
        ),
        80.9,
    ),
]


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_value_forced_scalar(convective_cooling, expected):
    """Test the _value_forced method of ConvectiveCooling."""
    Tf = 70.0
    Td = 60.0
    vm = 1.0

    result = convective_cooling._value_forced(Tf, Td, vm)

    assert np.allclose(result, expected, rtol=0.002)


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_value_forced_array(convective_cooling, expected):
    """Test the _value_forced method of ConvectiveCooling."""
    Tf = np.array([70.0, 70.0])
    Td = np.array([60.0, 60.0])
    vm = np.array([1.0, 1.0])

    result = convective_cooling._value_forced(Tf, Td, vm)

    assert np.allclose(result, expected, rtol=0.002)


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_value_natural_scalar(convective_cooling, expected):
    """Test the _value_natural method of ConvectiveCooling."""
    Td = 60.0
    vm = 1.0

    result = convective_cooling._value_natural(Td, vm)

    expected_result = (
        3.645
        * np.sqrt(vm)
        * convective_cooling.outer_diameter**0.75
        * np.sign(Td)
        * np.abs(Td) ** 1.25
    )
    assert np.allclose(result, expected_result, rtol=0.002)


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_value_natural_array(convective_cooling, expected):
    """Test the _value_natural method of ConvectiveCooling."""
    Td = np.array([60.0, 60.0])
    vm = np.array([1.0, 1.0])

    result = convective_cooling._value_natural(Td, vm)

    expected_result = (
        3.645
        * np.sqrt(vm)
        * convective_cooling.outer_diameter**0.75
        * np.sign(Td)
        * np.abs(Td) ** 1.25
    )
    assert np.allclose(result, expected_result, rtol=0.002)


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_convective_cooling_value_scalar(convective_cooling, expected):
    """Test the value method of ConvectiveCooling."""
    conductor_temperature = 100.0

    result = convective_cooling.value(conductor_temperature)

    expected_result = np.maximum(
        convective_cooling._value_forced(
            0.5 * (conductor_temperature + convective_cooling.ambient_temp),
            conductor_temperature - convective_cooling.ambient_temp,
            convective_cooling.air_density(
                0.5 * (conductor_temperature + convective_cooling.ambient_temp),
                convective_cooling.altitude,
            ),
        ),
        convective_cooling._value_natural(
            conductor_temperature - convective_cooling.ambient_temp,
            convective_cooling.air_density(
                0.5 * (conductor_temperature + convective_cooling.ambient_temp),
                convective_cooling.altitude,
            ),
        ),
    )
    assert np.allclose(result, expected_result, rtol=0.002)


@pytest.mark.parametrize(
    "convective_cooling, expected",
    convective_cooling_instances,
    ids=[
        "ConvectiveCooling with arrays",
        "ConvectiveCooling with scalars",
    ],
)
def test_convective_cooling_value_array(convective_cooling, expected):
    """Test the value method of ConvectiveCooling."""
    conductor_temperature = np.array([60.3, 100.0])

    result = convective_cooling.value(conductor_temperature)

    expected_result = np.maximum(
        convective_cooling._value_forced(
            0.5 * (conductor_temperature + convective_cooling.ambient_temp),
            conductor_temperature - convective_cooling.ambient_temp,
            convective_cooling.air_density(
                0.5 * (conductor_temperature + convective_cooling.ambient_temp),
                convective_cooling.altitude,
            ),
        ),
        convective_cooling._value_natural(
            conductor_temperature - convective_cooling.ambient_temp,
            convective_cooling.air_density(
                0.5 * (conductor_temperature + convective_cooling.ambient_temp),
                convective_cooling.altitude,
            ),
        ),
    )
    assert np.allclose(result, expected_result, rtol=0.002)
