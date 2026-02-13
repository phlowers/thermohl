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
    temp_low_c = 20.0
    temp_high_c = 80.0
    linear_resistance_temp_low_ohm_m = 0.1
    linear_resistance_temp_high_ohm_m = 0.2
    expected = (
        linear_resistance_temp_high_ohm_m - linear_resistance_temp_low_ohm_m
    ) / (temp_high_c - temp_low_c)

    result = JouleHeating._c(
        temp_low_c,
        temp_high_c,
        linear_resistance_temp_low_ohm_m,
        linear_resistance_temp_high_ohm_m,
    )

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_array():
    temp_low_c = np.array([20.0, 30.0])
    temp_high_c = np.array([80.0, 90.0])
    linear_resistance_temp_low_ohm_m = np.array([0.1, 0.15])
    linear_resistance_temp_high_ohm_m = np.array([0.2, 0.25])
    expected = (
        linear_resistance_temp_high_ohm_m - linear_resistance_temp_low_ohm_m
    ) / (temp_high_c - temp_low_c)

    result = JouleHeating._c(
        temp_low_c,
        temp_high_c,
        linear_resistance_temp_low_ohm_m,
        linear_resistance_temp_high_ohm_m,
    )

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_mixed():
    temp_low_c = 20.0
    temp_high_c = np.array([80.0, 90.0])
    linear_resistance_temp_low_ohm_m = 0.1
    linear_resistance_temp_high_ohm_m = np.array([0.2, 0.25])
    expected = (
        linear_resistance_temp_high_ohm_m - linear_resistance_temp_low_ohm_m
    ) / (temp_high_c - temp_low_c)

    result = JouleHeating._c(
        temp_low_c,
        temp_high_c,
        linear_resistance_temp_low_ohm_m,
        linear_resistance_temp_high_ohm_m,
    )

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


joule_heating_instances = [
    JouleHeating(
        current_a=np.array([100.0]),
        temp_low_c=np.array([20.0, 25.0]),
        temp_high_c=np.array([80.0, 85.0]),
        linear_resistance_temp_low_ohm_m=np.array([0.1, 0.15]),
        linear_resistance_temp_high_ohm_m=np.array([0.2, 0.25]),
    ),
    JouleHeating(
        current_a=100.0,
        temp_low_c=20.0,
        temp_high_c=80.0,
        linear_resistance_temp_low_ohm_m=0.1,
        linear_resistance_temp_high_ohm_m=0.2,
    ),
    JouleHeating(
        current_a=100.0,
        temp_low_c=np.array([20.0, 25.0]),
        temp_high_c=np.array([80.0, 85.0]),
        linear_resistance_temp_low_ohm_m=np.array([0.1, 0.15]),
        linear_resistance_temp_high_ohm_m=np.array([0.2, 0.25]),
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
    T = 50.0
    expected = joule_heating.dc_resistance_low_c + joule_heating.temp_coeff_linear * (
        T - joule_heating.temp_low_c
    )

    result = joule_heating._rdc(T)

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
    T = np.array([30.0, 40.0])
    expected = joule_heating.dc_resistance_low_c + joule_heating.temp_coeff_linear * (
        T - joule_heating.temp_low_c
    )

    result = joule_heating._rdc(T)

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
    T = 50.0
    expected = (
        joule_heating.dc_resistance_low_c
        + joule_heating.temp_coeff_linear * (T - joule_heating.temp_low_c)
    ) * joule_heating.current_a**2

    result = joule_heating.value(T)

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
    T = np.array([30.0, 40.0])
    expected = (
        joule_heating.dc_resistance_low_c
        + joule_heating.temp_coeff_linear * (T - joule_heating.temp_low_c)
    ) * joule_heating.current_a**2

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_array_temperature_different_shape_should_throw_error():
    current_a = 100.0
    temp_low_c = np.array([20.0, 25.0])
    temp_high_c = np.array([80.0, 85.0])
    linear_resistance_temp_low_ohm_m = np.array([0.1, 0.15])
    linear_resistance_temp_high_ohm_m = np.array([0.2, 0.25])
    joule_heating = JouleHeating(
        current_a,
        temp_low_c,
        temp_high_c,
        linear_resistance_temp_low_ohm_m,
        linear_resistance_temp_high_ohm_m,
    )
    T = np.array([30.0, 40.0, 50.0])

    with pytest.raises(ValueError):
        joule_heating.value(T)


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
    expected = joule_heating.temp_coeff_linear * joule_heating.current_a**2

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
        * joule_heating.current_a**2
        * np.ones_like(conductor_temperature)
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
