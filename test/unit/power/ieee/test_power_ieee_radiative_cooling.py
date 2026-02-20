# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl.power.ieee import RadiativeCooling
from thermohl import solver


def set_default_values_scalar():
    dic = solver.default_values()
    dic["ambient_temperature"] = 40.0
    dic["outer_diameter"] = 28.14 * 1.0e-03
    dic["emissivity"] = 0.8
    return dic


def set_default_values_array():
    dic = solver.default_values()
    dic["ambient_temperature"] = np.array([25.0, 40.0])
    dic["outer_diameter"] = np.array([24.83 * 1.0e-03, 28.14 * 1.0e-03])
    dic["emissivity"] = np.array([0.9, 0.8])
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
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * (
            ((temp + 273.0) / 100.0) ** 4
            - ((radiative_cooling.ambient_temp + 273.0) / 100.0) ** 4
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
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * (
            ((temp + 273.0) / 100.0) ** 4
            - ((radiative_cooling.ambient_temp + 273.0) / 100.0) ** 4
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
        4.0
        * 1.78e-07
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * temp**3
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
        4.0
        * 1.78e-07
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * temp**3
    )

    assert np.allclose(result, expected_result, rtol=0.001)
