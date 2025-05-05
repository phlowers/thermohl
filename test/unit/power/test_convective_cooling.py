# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl import solver
from thermohl.power.ieee import ConvectiveCooling


def set_default_values_scalar():
    dic = solver.default_values()
    dic["ws"] = 0.61
    dic["wa"] = 0.0
    dic["epsilon"] = 0.8
    dic["alpha"] = 0.8
    dic["Ta"] = 40.0
    dic["THigh"] = 75.0
    dic["TLow"] = 25.0
    dic["RDCHigh"] = 8.688e-05
    dic["RDCLow"] = 7.283e-05
    dic["azm"] = 90.0
    dic["lat"] = 30.0
    dic["tb"] = 0.0
    dic["alt"] = 0.0
    dic["D"] = 28.14 * 1.0e-03
    dic["d"] = 10.4 * 1.0e-03
    dic["month"] = 6
    dic["day"] = 10
    dic["hour"] = 11.0
    return dic


def set_default_values_array():
    dic = solver.default_values()
    dic["ws"] = np.array([0.61, 0.83])
    dic["wa"] = np.array([0.0, 42.1])
    dic["epsilon"] = np.array([0.8, 0.9])
    dic["alpha"] = np.array([0.8, 0.9])
    dic["Ta"] = np.array([40.0, 32])
    dic["THigh"] = np.array([75.0, 70.0])
    dic["TLow"] = np.array([25.0, 20])
    dic["RDCHigh"] = np.array([8.688e-05, 8.688e-05])
    dic["RDCLow"] = np.array([7.283e-05, 7.283e-05])
    dic["azm"] = np.array([90.0, 90.0])
    dic["lat"] = np.array([30.0, 30.0])
    dic["tb"] = np.array([0.0, 0.0])
    dic["alt"] = np.array([0.0, 0.0])
    dic["D"] = np.array([28.14 * 1.0e-03, 28.14 * 1.0e-03])
    dic["d"] = np.array([10.4 * 1.0e-03, 10.4 * 1.0e-03])
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
        * convective_cooling.D**0.75
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
        * convective_cooling.D**0.75
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
    T = 100.0

    result = convective_cooling.value(T)

    expected_result = np.maximum(
        convective_cooling._value_forced(
            0.5 * (T + convective_cooling.Ta),
            T - convective_cooling.Ta,
            convective_cooling.rho(
                0.5 * (T + convective_cooling.Ta), convective_cooling.alt
            ),
        ),
        convective_cooling._value_natural(
            T - convective_cooling.Ta,
            convective_cooling.rho(
                0.5 * (T + convective_cooling.Ta), convective_cooling.alt
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
    T = np.array([60.3, 100.0])

    result = convective_cooling.value(T)

    expected_result = np.maximum(
        convective_cooling._value_forced(
            0.5 * (T + convective_cooling.Ta),
            T - convective_cooling.Ta,
            convective_cooling.rho(
                0.5 * (T + convective_cooling.Ta), convective_cooling.alt
            ),
        ),
        convective_cooling._value_natural(
            T - convective_cooling.Ta,
            convective_cooling.rho(
                0.5 * (T + convective_cooling.Ta), convective_cooling.alt
            ),
        ),
    )
    assert np.allclose(result, expected_result, rtol=0.002)
