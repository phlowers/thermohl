# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl import solver
from thermohl.solver import SolverType
from thermohl.solver.enums.heat_equation_type import HeatEquationType
from thermohl.solver.enums.variable_type import VariableType
from thermohl.solver.enums.power_type import PowerType
from thermohl.solver.base import _DEFPARAM as DP


_nprs = 123456


def _solvers(dic=None):
    return [
        solver._factory(
            dic=dic, heat_equation=HeatEquationType.WITH_ONE_TEMPERATURE, model=m
        )
        for m in [
            SolverType.SOLVER_RTE,
            SolverType.SOLVER_CIGRE,
            SolverType.SOLVER_IEEE,
            SolverType.SOLVER_OLLA,
        ]
    ]


def test_balance():
    tol = 1.0e-09
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
        transit=np.random.uniform(40.0, 4000.0, N),
        core_diameter=np.random.randint(2, size=N)
        * solver.default_values()["core_diameter"],
    )

    for s in _solvers(dic):
        df = s.steady_temperature(
            return_err=True, return_power=True, tol=tol, maxiter=64
        )
        assert np.all(df[VariableType.ERROR] < tol)
        bl = np.abs(
            df[PowerType.JOULE]
            + df[PowerType.SOLAR]
            - df[PowerType.CONVECTION]
            - df[PowerType.RADIATION]
            - df[PowerType.RAIN]
        )
        atol = np.maximum(
            np.abs(
                s.balance(df[VariableType.TEMPERATURE] + 0.5 * df[VariableType.ERROR])
            ),
            np.abs(
                s.balance(df[VariableType.TEMPERATURE] - 0.5 * df[VariableType.ERROR])
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
        df = s.steady_intensity(
            max_conductor_temperature=100.0,
            return_err=True,
            return_power=True,
            tol=1.0e-09,
            maxiter=64,
        )
        bl = (
            df[PowerType.JOULE]
            + df[PowerType.SOLAR]
            - df[PowerType.CONVECTION]
            - df[PowerType.RADIATION]
            - df[PowerType.RAIN]
        )
        assert np.allclose(bl, 0.0, atol=1.0e-06)
        s.args[VariableType.TRANSIT.value] = df[VariableType.TRANSIT].values
        s.update()
        dg = s.steady_temperature(
            return_err=True, return_power=True, tol=1.0e-09, maxiter=64
        )
        assert np.allclose(dg[VariableType.TEMPERATURE].values, 100.0)


def test_steady_intensity_hot_weather():
    ambient_temperature = np.array([30.0, 35.0, 40.0, 45.0, 50.0])

    solver_1t = solver.ieee(
        dic={
            "ambient_temperature": ambient_temperature,
        },
        heat_equation=HeatEquationType.WITH_ONE_TEMPERATURE,
    )

    # Here some ambient temperatures are above the maximum conductor temperature,
    # so there's no solution - the solver should raise a ValueError
    with pytest.raises(ValueError):
        solver_1t.steady_intensity(
            max_conductor_temperature=45,
            Imin=np.ones_like(ambient_temperature) * DP.imin,
            Imax=np.ones_like(ambient_temperature) * DP.imax,
        )
