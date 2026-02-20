# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl.power.ieee import JouleHeating


def test_c_scalar():
    temp_low = 20.0
    temp_high = 80.0
    linear_resistance_temp_low = 0.1
    linear_resistance_temp_high = 0.2
    expected = (linear_resistance_temp_high - linear_resistance_temp_low) / (
        temp_high - temp_low
    )

    result = JouleHeating._c(
        temp_low,
        temp_high,
        linear_resistance_temp_low,
        linear_resistance_temp_high,
    )

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_array():
    temp_low = np.array([20.0, 30.0])
    temp_high = np.array([80.0, 90.0])
    linear_resistance_temp_low = np.array([0.1, 0.15])
    linear_resistance_temp_high = np.array([0.2, 0.25])
    expected = (linear_resistance_temp_high - linear_resistance_temp_low) / (
        temp_high - temp_low
    )

    result = JouleHeating._c(
        temp_low,
        temp_high,
        linear_resistance_temp_low,
        linear_resistance_temp_high,
    )

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_mixed():
    temp_low = 20.0
    temp_high = np.array([80.0, 90.0])
    linear_resistance_temp_low = 0.1
    linear_resistance_temp_high = np.array([0.2, 0.25])
    expected = (linear_resistance_temp_high - linear_resistance_temp_low) / (
        temp_high - temp_low
    )

    result = JouleHeating._c(
        temp_low,
        temp_high,
        linear_resistance_temp_low,
        linear_resistance_temp_high,
    )

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


joule_heating_instances = [
    JouleHeating(
        transit=np.array([100.0]),
        temp_low=np.array([20.0, 25.0]),
        temp_high=np.array([80.0, 85.0]),
        linear_resistance_temp_low=np.array([0.1, 0.15]),
        linear_resistance_temp_high=np.array([0.2, 0.25]),
    ),
    JouleHeating(
        transit=100.0,
        temp_low=20.0,
        temp_high=80.0,
        linear_resistance_temp_low=0.1,
        linear_resistance_temp_high=0.2,
    ),
    JouleHeating(
        transit=100.0,
        temp_low=np.array([20.0, 25.0]),
        temp_high=np.array([80.0, 85.0]),
        linear_resistance_temp_low=np.array([0.1, 0.15]),
        linear_resistance_temp_high=np.array([0.2, 0.25]),
    ),
]


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_rdc_scalar_temperature(joule_heating):
    conductor_temperature = 50.0
    expected = (
        joule_heating.linear_resistance_temp_low
        + joule_heating.temp_coeff_linear
        * (conductor_temperature - joule_heating.temp_low)
    )

    result = joule_heating._rdc(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_rdc_array_temperature(joule_heating):
    conductor_temperature = np.array([30.0, 40.0])
    expected = (
        joule_heating.linear_resistance_temp_low
        + joule_heating.temp_coeff_linear
        * (conductor_temperature - joule_heating.temp_low)
    )

    result = joule_heating._rdc(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_value_scalar_temperature(joule_heating):
    conductor_temperature = 50.0
    expected = (
        joule_heating.linear_resistance_temp_low
        + joule_heating.temp_coeff_linear
        * (conductor_temperature - joule_heating.temp_low)
    ) * joule_heating.transit**2

    result = joule_heating.value(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_value_array_temperature(joule_heating):
    conductor_temperature = np.array([30.0, 40.0])
    expected = (
        joule_heating.linear_resistance_temp_low
        + joule_heating.temp_coeff_linear
        * (conductor_temperature - joule_heating.temp_low)
    ) * joule_heating.transit**2

    result = joule_heating.value(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_array_temperature_different_shape_should_throw_error():
    transit = 100.0
    temp_low = np.array([20.0, 25.0])
    temp_high = np.array([80.0, 85.0])
    linear_resistance_temp_low = np.array([0.1, 0.15])
    linear_resistance_temp_high = np.array([0.2, 0.25])
    joule_heating = JouleHeating(
        transit,
        temp_low,
        temp_high,
        linear_resistance_temp_low,
        linear_resistance_temp_high,
    )
    conductor_temperature = np.array([30.0, 40.0, 50.0])

    with pytest.raises(ValueError):
        joule_heating.value(conductor_temperature)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_derivative_scalar_temperature(joule_heating):
    conductor_temperature = 50.0
    expected = joule_heating.temp_coeff_linear * joule_heating.transit**2

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
        "JouleHeating with mixed types",
    ],
)
def test_derivative_array_temperature(joule_heating):
    conductor_temperature = np.array([30.0, 40.0])
    expected = (
        joule_heating.temp_coeff_linear
        * joule_heating.transit**2
        * np.ones_like(conductor_temperature)
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
