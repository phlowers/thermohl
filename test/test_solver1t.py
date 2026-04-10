# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime
import pytest
import random
import numpy as np

from thermohl import solver
from thermohl.solver import ModelType
from thermohl.solver.entities import (
    HeatEquationType,
    VariableType,
    PowerType,
)
from thermohl.solver.parameters import DEFAULT_PARAMETERS as DP

_nprs = 123456


def _solvers(dic=None):
    return [
        solver._factory(
            dic=dic, heat_equation=HeatEquationType.ONE_TEMPERATURE, model=m
        )
        for m in [
            ModelType.RTE,
            ModelType.CIGRE,
            ModelType.IEEE,
            ModelType.OLLA,
        ]
    ]


def test_balance():
    tol = 1.0e-09
    random.seed(_nprs)
    np.random.seed(_nprs)
    N = 9999
    month = random.randint(1, 12)
    day = random.randint(1, 30)
    hour = random.randint(0, 23)
    dic = dict(
        latitude=np.random.uniform(42.0, 51.0, N),
        altitude=np.random.uniform(0.0, 1600.0, N),
        cable_azimuth=np.random.uniform(0.0, 360.0, N),
        datetime_utc=[datetime(2026, month, day, hour) for _ in range(N)],
        ambient_temperature=np.random.uniform(0.0, 30.0, N),
        wind_speed=np.random.uniform(0.0, 7.0, N),
        wind_azimuth=np.random.uniform(0.0, 90.0, N),
        transit=np.random.uniform(40.0, 4000.0, N),
        core_diameter=np.random.randint(2, size=N)
        * solver.default_values()["core_diameter"],
    )

    for s in _solvers(dic):
        steady_temperature = s.steady_temperature(
            return_err=True, return_power=True, tol=tol, maxiter=64
        )
        assert np.all(steady_temperature[VariableType.ERROR.value] < tol)
        bl = np.abs(
            steady_temperature[PowerType.JOULE.value]
            + steady_temperature[PowerType.SOLAR.value]
            - steady_temperature[PowerType.CONVECTION.value]
            - steady_temperature[PowerType.RADIATION.value]
            - steady_temperature[PowerType.RAIN.value]
        )
        atol = np.maximum(
            np.abs(
                s.balance(
                    steady_temperature[VariableType.TEMPERATURE.value]
                    + 0.5 * steady_temperature[VariableType.ERROR.value]
                )
            ),
            np.abs(
                s.balance(
                    steady_temperature[VariableType.TEMPERATURE.value]
                    - 0.5 * steady_temperature[VariableType.ERROR.value]
                )
            ),
        )
        assert np.allclose(bl, 0.0, atol=atol)


def test_consistency():
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        latitude=np.random.uniform(42.0, 51.0, N),
        altitude=np.random.uniform(0.0, 1600.0, N),
        cable_azimuth=np.random.uniform(0.0, 360.0, N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        ambient_temperature=np.random.uniform(0.0, 30.0, N),
        wind_speed=np.random.uniform(0.0, 7.0, N),
        wind_azimuth=np.random.uniform(0.0, 90.0, N),
        core_diameter=np.random.randint(2, size=N)
        * solver.default_values()["core_diameter"],
    )

    for s in _solvers(dic):
        steady_intensity = s.steady_intensity(
            max_conductor_temperature=100.0,
            return_err=True,
            return_power=True,
            tol=1.0e-09,
            maxiter=64,
        )
        bl = (
            steady_intensity[PowerType.JOULE.value]
            + steady_intensity[PowerType.SOLAR.value]
            - steady_intensity[PowerType.CONVECTION.value]
            - steady_intensity[PowerType.RADIATION.value]
            - steady_intensity[PowerType.RAIN.value]
        )
        assert np.allclose(bl, 0.0, atol=1.0e-06)
        s.args[VariableType.TRANSIT.value] = steady_intensity[
            VariableType.TRANSIT.value
        ]
        s.update()
        steady_temperature = s.steady_temperature(
            return_err=True, return_power=True, tol=1.0e-09, maxiter=64
        )
        assert np.allclose(steady_temperature[VariableType.TEMPERATURE.value], 100.0)


def test_steady_intensity_hot_weather():
    ambient_temperature = np.array([30.0, 35.0, 40.0, 45.0, 50.0])

    solver_1t = solver.ieee(
        dic={
            "ambient_temperature": ambient_temperature,
        },
        heat_equation=HeatEquationType.ONE_TEMPERATURE,
    )

    # Here some ambient temperatures are above the maximum conductor temperature,
    # so there's no solution - the solver should raise a ValueError
    with pytest.raises(ValueError):
        solver_1t.steady_intensity(
            max_conductor_temperature=45,
            Imin=np.ones_like(ambient_temperature) * DP.imin,
            Imax=np.ones_like(ambient_temperature) * DP.imax,
        )
