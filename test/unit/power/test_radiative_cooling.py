# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl.power.radiative_cooling import RadiativeCoolingBase

radiative_cooling_instances = [
    RadiativeCoolingBase(
        ambient_temperature=np.array([25.0, 15.2]),
        outer_diameter=np.array([1, 1.5]),
        emissivity=np.array([0.9, 0.8]),
    ),
    RadiativeCoolingBase(ambient_temperature=25, outer_diameter=1, emissivity=0.9),
]


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_celsius2kelvin_scalar(radiative_cooling):
    assert radiative_cooling._celsius2kelvin(0) == 273.15
    assert radiative_cooling._celsius2kelvin(100) == 373.15


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_celsius2kelvin_array(radiative_cooling):
    temp_celsius = np.array([0, 100, -273.15])

    temp_kelvin = radiative_cooling._celsius2kelvin(temp_celsius)

    np.testing.assert_array_equal(temp_kelvin, np.array([273.15, 373.15, 0]))


def test_celsius2kelvin_with_custom_zerok():
    radiative_cooling = RadiativeCoolingBase(
        ambient_temperature=25, outer_diameter=1, emissivity=0.9, zerok=0
    )

    assert radiative_cooling._celsius2kelvin(0) == 0
    assert radiative_cooling._celsius2kelvin(100) == 100


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_value_scalar(radiative_cooling):
    conductor_temp = 100

    expected_value = (
        np.pi
        * radiative_cooling.stefan_boltzmann
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * (
            (radiative_cooling._celsius2kelvin(conductor_temp)) ** 4
            - radiative_cooling.ambient_temperature**4
        )
    )

    assert np.allclose(expected_value, radiative_cooling.value(conductor_temp))


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_value_array(radiative_cooling):
    conductor_temp = np.array([50, 100])

    expected_value = (
        np.pi
        * radiative_cooling.stefan_boltzmann
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * (
            (radiative_cooling._celsius2kelvin(conductor_temp)) ** 4
            - radiative_cooling.ambient_temperature**4
        )
    )

    np.testing.assert_array_almost_equal(
        radiative_cooling.value(conductor_temp), expected_value
    )


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_derivative_scalar(radiative_cooling):
    conductor_temperature = 100

    expected = (
        4.0
        * np.pi
        * radiative_cooling.stefan_boltzmann
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * conductor_temperature**3
    )
    assert np.allclose(expected, radiative_cooling.derivative(conductor_temperature))


@pytest.mark.parametrize(
    "radiative_cooling",
    radiative_cooling_instances,
    ids=[
        "RadiativeCooling with arrays",
        "RadiativeCooling with scalars",
    ],
)
def test_derivative_array(radiative_cooling):
    # radiative_cooling = RadiativeCoolingBase(ambient_temperature=25, outer_diameter=1, emissivity=0.9, zerok=0)
    conductor_temperature = np.array([100, 200])

    expected = (
        4.0
        * np.pi
        * radiative_cooling.stefan_boltzmann
        * radiative_cooling.emissivity
        * radiative_cooling.outer_diameter
        * conductor_temperature**3
    )
    np.testing.assert_array_equal(
        radiative_cooling.derivative(conductor_temperature), expected
    )
