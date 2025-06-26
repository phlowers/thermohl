# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.ieee import JouleHeating


def test_c_scalar():
    TLow = 20.0
    THigh = 80.0
    RDCLow = 0.1
    RDCHigh = 0.2
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_array():
    TLow = np.array([20.0, 30.0])
    THigh = np.array([80.0, 90.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_c_mixed():
    TLow = 20.0
    THigh = np.array([80.0, 90.0])
    RDCLow = 0.1
    RDCHigh = np.array([0.2, 0.25])
    expected = (RDCHigh - RDCLow) / (THigh - TLow)

    result = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


joule_heating_instances = [
    JouleHeating(
        I=np.array([100.0]),
        TLow=np.array([20.0, 25.0]),
        THigh=np.array([80.0, 85.0]),
        RDCLow=np.array([0.1, 0.15]),
        RDCHigh=np.array([0.2, 0.25]),
    ),
    JouleHeating(
        I=100.0,
        TLow=20.0,
        THigh=80.0,
        RDCLow=0.1,
        RDCHigh=0.2,
    ),
    JouleHeating(
        I=100.0,
        TLow=np.array([20.0, 25.0]),
        THigh=np.array([80.0, 85.0]),
        RDCLow=np.array([0.1, 0.15]),
        RDCHigh=np.array([0.2, 0.25]),
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
    expected = joule_heating.RDCLow + joule_heating.c * (T - joule_heating.TLow)

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
    expected = joule_heating.RDCLow + joule_heating.c * (T - joule_heating.TLow)

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
        joule_heating.RDCLow + joule_heating.c * (T - joule_heating.TLow)
    ) * joule_heating.I**2

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
        joule_heating.RDCLow + joule_heating.c * (T - joule_heating.TLow)
    ) * joule_heating.I**2

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_value_array_temperature_different_shape_should_throw_error():
    I = 100.0
    TLow = np.array([20.0, 25.0])
    THigh = np.array([80.0, 85.0])
    RDCLow = np.array([0.1, 0.15])
    RDCHigh = np.array([0.2, 0.25])
    joule_heating = JouleHeating(I, TLow, THigh, RDCLow, RDCHigh)
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
    expected = joule_heating.c * joule_heating.I**2

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
        joule_heating.c * joule_heating.I**2 * np.ones_like(conductor_temperature)
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
