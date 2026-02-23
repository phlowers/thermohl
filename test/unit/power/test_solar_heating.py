# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl import sun
from thermohl.power import _SRad
from thermohl.power.ieee import SolarHeating


# Tests Classe SRad
@pytest.fixture
def srad():
    clean = [
        -4.22391e01,
        +6.38044e01,
        -1.9220e00,
        +3.46921e-02,
        -3.61118e-04,
        +1.94318e-06,
        -4.07608e-09,
    ]
    indus = [
        +5.31821e01,
        +1.4211e01,
        +6.6138e-01,
        -3.1658e-02,
        +5.4654e-04,
        -4.3446e-06,
        +1.3236e-08,
    ]
    return _SRad(clean, indus)


def test_srad_atmosphere_turbidity_scalar(srad):
    solar_altitude = 30.0
    turbidity = 0.5
    omt = 1.0 - turbidity
    A = omt * srad.clean[6] + turbidity * srad.indus[6]
    B = omt * srad.clean[5] + turbidity * srad.indus[5]
    C = omt * srad.clean[4] + turbidity * srad.indus[4]
    outer_diameter = omt * srad.clean[3] + turbidity * srad.indus[3]
    E = omt * srad.clean[2] + turbidity * srad.indus[2]
    F = omt * srad.clean[1] + turbidity * srad.indus[1]
    G = omt * srad.clean[0] + turbidity * srad.indus[0]
    expected = (
        A * solar_altitude**6
        + B * solar_altitude**5
        + C * solar_altitude**4
        + outer_diameter * solar_altitude**3
        + E * solar_altitude**2
        + F * solar_altitude**1
        + G
    )

    result = srad.atmosphere_turbidity(solar_altitude, turbidity)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_srad_atmosphere_turbidity_array(srad):
    solar_altitude = np.array([30.0, 40.0])
    turbidity = np.array([0.5, 0.7])
    omt = 1.0 - turbidity
    A = omt * srad.clean[6] + turbidity * srad.indus[6]
    B = omt * srad.clean[5] + turbidity * srad.indus[5]
    C = omt * srad.clean[4] + turbidity * srad.indus[4]
    outer_diameter = omt * srad.clean[3] + turbidity * srad.indus[3]
    E = omt * srad.clean[2] + turbidity * srad.indus[2]
    F = omt * srad.clean[1] + turbidity * srad.indus[1]
    G = omt * srad.clean[0] + turbidity * srad.indus[0]
    expected = (
        A * solar_altitude**6
        + B * solar_altitude**5
        + C * solar_altitude**4
        + outer_diameter * solar_altitude**3
        + E * solar_altitude**2
        + F * solar_altitude**1
        + G
    )

    result = srad.atmosphere_turbidity(solar_altitude, turbidity)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_srad_call_scalar(srad):
    latitude = 45.0
    altitude = 1000.0
    azimuth = 180.0
    turbidity = 0.5
    month = 6
    day = 21
    hour = 12.0
    sa = sun.solar_altitude(latitude, month, day, hour)
    sz = sun.solar_azimuth(latitude, month, day, hour)
    th = np.arccos(np.cos(sa) * np.cos(sz - azimuth))
    K = 1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude**2
    Q = srad.atmosphere_turbidity(np.rad2deg(sa), turbidity)
    expected = K * Q * np.sin(th)
    expected = np.where(expected > 0.0, expected, 0.0)

    result = srad(latitude, altitude, azimuth, turbidity, month, day, hour)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


def test_srad_call_array(srad):
    latitude = np.array([45.0, 50.0])
    altitude = np.array([1000.0, 2000.0])
    azimuth = np.array([180.0, 190.0])
    turbidity = np.array([0.5, 0.7])
    month = np.array([6, 7])
    day = np.array([21, 22])
    hour = np.array([12.0, 13.0])
    sa = sun.solar_altitude(latitude, month, day, hour)
    sz = sun.solar_azimuth(latitude, month, day, hour)
    th = np.arccos(np.cos(sa) * np.cos(sz - azimuth))
    K = 1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude**2
    Q = srad.atmosphere_turbidity(np.rad2deg(sa), turbidity)
    expected = K * Q * np.sin(th)
    expected = np.where(expected > 0.0, expected, 0.0)

    result = srad(latitude, altitude, azimuth, turbidity, month, day, hour)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


# Tests Class SolarHeating
solar_heating_instances = [
    SolarHeating(
        latitude=np.array([45.0, 50.0]),
        altitude=np.array([1000.0, 2000.0]),
        azimuth=np.array([180.0, 190.0]),
        turbidity=np.array([0.5, 0.7]),
        month=np.array([6, 7]),
        day=np.array([21, 22]),
        hour=np.array([12.0, 13.0]),
        outer_diameter=np.array([0.01, 0.02]),
        solar_absorptivity=np.array([0.9, 0.8]),
        srad=np.array([800.0, 900.0]),
    ),
    SolarHeating(
        latitude=45.0,
        altitude=1000.0,
        azimuth=180.0,
        turbidity=0.5,
        month=6,
        day=21,
        hour=12.0,
        outer_diameter=0.01,
        solar_absorptivity=0.9,
        srad=800.0,
    ),
]


@pytest.mark.parametrize(
    "solar_heating",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_value_scalar(solar_heating):
    conductor_temperature = 50.0
    expected = (
        solar_heating.solar_absorptivity
        * solar_heating.solar_irradiance
        * solar_heating.outer_diameter
    )

    result = solar_heating.value(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "solar_heating",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_value_array(solar_heating):
    conductor_temperature = np.array([50.0, 60.0])
    expected = (
        solar_heating.solar_absorptivity
        * solar_heating.solar_irradiance
        * solar_heating.outer_diameter
    )

    result = solar_heating.value(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "solar_heating",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_derivative_scalar(solar_heating):
    conductor_temperature = 50.0
    expected = 0.0

    result = solar_heating.derivative(conductor_temperature)

    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "solar_heating",
    solar_heating_instances,
    ids=[
        "SolarHeating with arrays",
        "SolarHeating with scalars",
    ],
)
def test_solar_heating_derivative_array(solar_heating):
    conductor_temperature = np.array([50.0, 60.0])
    expected = np.array([0.0, 0.0])

    result = solar_heating.derivative(conductor_temperature)

    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
