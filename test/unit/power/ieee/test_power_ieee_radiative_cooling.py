# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl import solver
from thermohl.power.ieee import RadiativeCooling


def set_default_values_scalar():
    dic = solver.default_values()
    dic["Ta"] = 40.0
    dic["D"] = 28.14 * 1.0e-03
    dic["epsilon"] = 0.8
    return dic


def set_default_values_array():
    dic = solver.default_values()
    dic["Ta"] = np.array([25.0, 40.0])
    dic["D"] = np.array([24.83 * 1.0e-03, 28.14 * 1.0e-03])
    dic["epsilon"] = np.array([0.9, 0.8])
    return dic


radiative_cooling_instances = [
    RadiativeCooling(**set_default_values_array()),
    RadiativeCooling(
        **set_default_values_scalar(),
    ),
]


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_radiative_cooling_value_temperature_scalar(radiative_cooling):
    temp = 100.0
    expected_result = (
        17.8
        * radiative_cooling.epsilon
        * radiative_cooling.D
        * (
            ((temp + 273.0) / 100.0) ** 4
            - ((radiative_cooling.Ta + 273.0) / 100.0) ** 4
        )
    )

    result = radiative_cooling.value(temp)

    assert np.allclose(result, expected_result, rtol=0.001)


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_radiative_cooling_value_temperature_array(radiative_cooling):
    temp = np.array([65.3, 100.0])
    expected_result = (
        17.8
        * radiative_cooling.epsilon
        * radiative_cooling.D
        * (
            ((temp + 273.0) / 100.0) ** 4
            - ((radiative_cooling.Ta + 273.0) / 100.0) ** 4
        )
    )

    result = radiative_cooling.value(temp)

    assert np.allclose(result, expected_result, rtol=0.001)


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_radiative_cooling_derivative_temperature_scalar(radiative_cooling):
    temp = 100.0
    result = radiative_cooling.derivative(temp)

    expected_result = (
        4.0 * 1.78e-07 * radiative_cooling.epsilon * radiative_cooling.D * temp**3
    )

    assert np.allclose(result, expected_result, rtol=0.001)


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_radiative_cooling_derivative_temperature_array(radiative_cooling):
    temp = np.array([65.3, 100.0])
    result = radiative_cooling.derivative(temp)

    expected_result = (
        4.0 * 1.78e-07 * radiative_cooling.epsilon * radiative_cooling.D * temp**3
    )

    assert np.allclose(result, expected_result, rtol=0.001)
