# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl.power.rte import JouleHeating

joule_heating_instances = [
    JouleHeating(
        transit=np.array([10.0]),
        outer_diameter=np.array([0.01]),
        core_diameter=np.array([0.005]),
        outer_area=np.array([0.0001]),
        core_area=np.array([0.00005]),
        magnetic_coeff=np.array([1.0]),
        magnetic_coeff_per_a=np.array([0.1]),
        temperature_coeff_linear=np.array([0.004]),
        temperature_coeff_quadratic=np.array([0.0001]),
        linear_resistance_dc_20c=np.array([0.02]),
        reference_temperature=20.0,
        frequency=50.0,
    ),
    JouleHeating(
        transit=10.0,
        outer_diameter=0.01,
        core_diameter=0.005,
        outer_area=0.0001,
        core_area=0.00005,
        magnetic_coeff=1.0,
        magnetic_coeff_per_a=0.1,
        temperature_coeff_linear=0.004,
        temperature_coeff_quadratic=0.0001,
        linear_resistance_dc_20c=0.02,
        reference_temperature=20.0,
        frequency=50.0,
    ),
]


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_rdc(joule_heating):
    conductor_temperature = np.array([30.0])
    expected_rdc = 0.021

    result = joule_heating._rdc(conductor_temperature)

    np.testing.assert_allclose(result, expected_rdc, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_ks(joule_heating):
    conductor_temperature = np.array([30.0])

    rdc = joule_heating._rdc(conductor_temperature)
    expected_ks = 1.0

    result = joule_heating._ks(rdc)

    np.testing.assert_allclose(result, expected_ks, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_kem(joule_heating):
    outer_area = np.array([0.0001])
    core_area = np.array([0.00005])
    magnetic_coeff = np.array([1.0])
    magnetic_coeff_per_a = np.array([0.1])
    expected_kem = 1.02

    result = joule_heating._kem(
        outer_area, core_area, magnetic_coeff, magnetic_coeff_per_a
    )

    np.testing.assert_allclose(result, expected_kem, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_joule_heating_value(joule_heating):
    conductor_temperature = np.array([30.0])
    expected_value = 2.1420

    result = joule_heating.value(conductor_temperature)

    np.testing.assert_allclose(result, expected_value, rtol=1e-5)
