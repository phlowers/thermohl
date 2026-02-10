# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.cigre import JouleHeating

joule_heating_instances = [
    JouleHeating(
        transit=np.array([100.0, 150.0, 200.0]),
        magnetic_coeff=np.array([1.0, 1.0, 1.0]),
        temperature_coeff_linear=np.array([0.004, 0.004, 0.004]),
        dc_resistance_20c_ohm_m=np.array([0.1, 0.1, 0.1]),
        T20=np.array([20.0, 18.0, 22.0]),
    ),
    JouleHeating(
        transit=100.0,
        magnetic_coeff=1.0,
        temperature_coeff_linear=0.004,
        dc_resistance_20c_ohm_m=0.1,
        T20=20.0,
    ),
]


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
    ],
)
def test_joule_heating_value_scalar(joule_heating):
    T = 25.0
    expected = (
        joule_heating.magnetic_coeff
        * joule_heating.dc_resistance_20c
        * (1.0 + joule_heating.temp_coeff_linear * (T - joule_heating.reference_temp_c))
        * joule_heating.current_a**2
    )

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
    ],
)
def test_joule_heating_value_array(joule_heating):
    T = np.array([25.0, 30.0, 35.0])
    expected = (
        joule_heating.magnetic_coeff
        * joule_heating.dc_resistance_20c
        * (1.0 + joule_heating.temp_coeff_linear * (T - joule_heating.reference_temp_c))
        * joule_heating.current_a**2
    )

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_joule_heating_value_mismatched_array_sizes_should_raise_error():
    transit = np.array([100.0, 150.0])
    magnetic_coeff = np.array([1.0, 1.0, 1.0])
    temperature_coeff_linear = np.array([0.004, 0.004])
    dc_resistance_20c_ohm_m = np.array([0.1, 0.1])
    T20 = np.array([20.0, 20.0])
    T = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        joule_heating = JouleHeating(
            transit,
            magnetic_coeff,
            temperature_coeff_linear,
            dc_resistance_20c_ohm_m,
            T20,
        )
        joule_heating.value(T)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
    ],
)
def test_joule_heating_derivative_scalar(joule_heating):
    conductor_temperature = 25.0
    expected = (
        joule_heating.magnetic_coeff
        * joule_heating.dc_resistance_20c
        * joule_heating.temp_coeff_linear
        * joule_heating.current_a**2
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=[
        "JouleHeating with arrays",
        "JouleHeating with scalars",
    ],
)
def test_joule_heating_derivative_array(joule_heating):
    conductor_temperature = np.array([25.0, 30.0, 35.0])
    expected = (
        joule_heating.magnetic_coeff
        * joule_heating.dc_resistance_20c
        * joule_heating.temp_coeff_linear
        * joule_heating.current_a**2
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_joule_heating_derivative_mismatched_array_sizes_should_raise_error():
    transit = np.array([100.0, 150.0])
    magnetic_coeff = np.array([1.0, 1.0, 1.0])
    temperature_coeff_linear = np.array([0.004, 0.004])
    dc_resistance_20c_ohm_m = np.array([0.1, 0.1])
    T20 = np.array([20.0, 20.0])
    conductor_temperature = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        joule_heating = JouleHeating(
            transit,
            magnetic_coeff,
            temperature_coeff_linear,
            dc_resistance_20c_ohm_m,
            T20,
        )
        joule_heating.derivative(conductor_temperature)
