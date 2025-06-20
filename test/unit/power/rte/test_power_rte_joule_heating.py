# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.rte import JouleHeating

joule_heating_instances = [
    JouleHeating(
        I=np.array([10.0]),
        D=np.array([0.01]),
        d=np.array([0.005]),
        A=np.array([0.0001]),
        a=np.array([0.00005]),
        km=np.array([1.0]),
        ki=np.array([0.1]),
        kl=np.array([0.004]),
        kq=np.array([0.0001]),
        RDC20=np.array([0.02]),
        T20=20.0,
        f=50.0,
    ),
    JouleHeating(
        I=10.0,
        D=0.01,
        d=0.005,
        A=0.0001,
        a=0.00005,
        km=1.0,
        ki=0.1,
        kl=0.004,
        kq=0.0001,
        RDC20=0.02,
        T20=20.0,
        f=50.0,
    ),
]


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_rdc(joule_heating):
    T = np.array([30.0])
    expected_rdc = 0.021

    result = joule_heating._rdc(T)

    np.testing.assert_allclose(result, expected_rdc, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_ks(joule_heating):
    T = np.array([30.0])

    rdc = joule_heating._rdc(T)
    expected_ks = 1.0

    result = joule_heating._ks(rdc)

    np.testing.assert_allclose(result, expected_ks, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_kem(joule_heating):
    A = np.array([0.0001])
    a = np.array([0.00005])
    km = np.array([1.0])
    ki = np.array([0.1])
    expected_kem = 1.02

    result = joule_heating._kem(A, a, km, ki)

    np.testing.assert_allclose(result, expected_kem, rtol=1e-5)


@pytest.mark.parametrize(
    "joule_heating",
    joule_heating_instances,
    ids=["JouleHeating with arrays", "JouleHeating with scalars"],
)
def test_joule_heating_value(joule_heating):
    T = np.array([30.0])
    expected_value = 2.1420

    result = joule_heating.value(T)

    np.testing.assert_allclose(result, expected_value, rtol=1e-5)
