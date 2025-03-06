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
        I=np.array([100.0, 150.0, 200.0]),
        km=np.array([1.0, 1.0, 1.0]),
        kl=np.array([0.004, 0.004, 0.004]),
        RDC20=np.array([0.1, 0.1, 0.1]),
        T20=np.array([20.0, 18.0, 22.0]),
    ),
    JouleHeating(I=100.0, km=1.0, kl=0.004, RDC20=0.1, T20=20.0),
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
        joule_heating.km
        * joule_heating.RDC20
        * (1.0 + joule_heating.kl * (T - joule_heating.T20))
        * joule_heating.I**2
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
        joule_heating.km
        * joule_heating.RDC20
        * (1.0 + joule_heating.kl * (T - joule_heating.T20))
        * joule_heating.I**2
    )

    result = joule_heating.value(T)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_joule_heating_value_mismatched_array_sizes_should_raise_error():
    I = np.array([100.0, 150.0])
    km = np.array([1.0, 1.0, 1.0])
    kl = np.array([0.004, 0.004])
    RDC20 = np.array([0.1, 0.1])
    T20 = np.array([20.0, 20.0])
    T = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        joule_heating = JouleHeating(I, km, kl, RDC20, T20)
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
        joule_heating.km * joule_heating.RDC20 * joule_heating.kl * joule_heating.I**2
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
        joule_heating.km * joule_heating.RDC20 * joule_heating.kl * joule_heating.I**2
    )

    result = joule_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_joule_heating_derivative_mismatched_array_sizes_should_raise_error():
    I = np.array([100.0, 150.0])
    km = np.array([1.0, 1.0, 1.0])
    kl = np.array([0.004, 0.004])
    RDC20 = np.array([0.1, 0.1])
    T20 = np.array([20.0, 20.0])
    conductor_temperature = np.array([25.0, 30.0])
    with pytest.raises(ValueError):
        joule_heating = JouleHeating(I, km, kl, RDC20, T20)
        joule_heating.derivative(conductor_temperature)
