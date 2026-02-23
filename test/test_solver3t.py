# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl import solver
from thermohl.solver import HeatEquationType
from thermohl.solver.enums.solver_type import SolverType
from thermohl.solver.enums.variable_type import VariableType
from thermohl.solver.enums.temperature_location import TemperatureLocation
from thermohl.solver.enums.cable_location import CableLocation

_nprs = 123456


def _solvers(dic=None):
    return [
        solver._factory(dic=dic, heat_equation=heat_equation, model=m)
        for heat_equation in [
            HeatEquationType.WITH_THREE_TEMPERATURES,
            HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
        ]
        for m in [
            SolverType.SOLVER_RTE,
            SolverType.SOLVER_CIGRE,
            SolverType.SOLVER_IEEE,
            SolverType.SOLVER_OLLA,
        ]
    ]


def test_balance():
    # NB : this one fails only with 'cigre' powers

    tol = 1.0e-09
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        latitude=np.random.uniform(42.0, 51.0, N),
        altitude=np.random.uniform(0.0, 1600.0, N),
        azimuth=np.random.uniform(0.0, 360.0, N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        ambient_temperature=np.random.uniform(0.0, 30.0, N),
        wind_speed=np.random.uniform(0.0, 7.0, N),
        wind_angle=np.random.uniform(0.0, 90.0, N),
        transit=np.random.uniform(40.0, 4000.0, N),
        core_diameter=np.random.randint(2, size=N)
        * solver.default_values()["core_diameter"],
    )

    for s in _solvers(dic):
        # compute guess with 1t solver
        s1 = solver._factory(
            dic=dic,
            heat_equation=HeatEquationType.WITH_ONE_TEMPERATURE,
            model=SolverType.SOLVER_IEEE,
        )
        t1 = s1.steady_temperature(
            tol=2.0, maxiter=16, return_err=False, return_power=False
        )
        t1 = t1[VariableType.TEMPERATURE].values
        # 3t solve
        df = s.steady_temperature(
            surface_temperature_guess=t1,
            core_temperature_guess=t1,
            return_err=True,
            return_power=True,
            tol=tol,
            maxiter=64,
        )
        # checks
        assert np.all(df[VariableType.ERROR] < tol)
        assert np.allclose(
            s.balance(
                surface_temperature=df[TemperatureLocation.SURFACE],
                core_temperature=df[TemperatureLocation.CORE],
            ).values,
            0.0,
            atol=tol,
        )
        assert np.allclose(
            s.morgan(
                surface_temperature=df[TemperatureLocation.SURFACE],
                core_temperature=df[TemperatureLocation.CORE],
            ).values,
            0.0,
            atol=tol,
        )


def test_consistency():
    tol = 1.0e-09
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        latitude=np.random.uniform(42.0, 51.0, N),
        altitude=np.random.uniform(0.0, 1600.0, N),
        azimuth=np.random.uniform(0.0, 360.0, N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        ambient_temperature=np.random.uniform(0.0, 30.0, N),
        wind_speed=np.random.uniform(0.0, 7.0, N),
        wind_angle=np.random.uniform(0.0, 90.0, N),
        core_diameter=np.random.randint(2, size=N)
        * solver.default_values()["core_diameter"],
    )

    for s in _solvers(dic):
        d = {
            CableLocation.SURFACE: TemperatureLocation.SURFACE,
            CableLocation.AVERAGE: TemperatureLocation.AVERAGE,
            CableLocation.CORE: TemperatureLocation.CORE,
        }

        for location, temperature_at_location in d.items():
            # solve intensity with different targets
            df = s.steady_intensity(
                max_conductor_temperature=100.0,
                target=location,
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            assert np.all(df[VariableType.ERROR] < tol)
            # set args intensity to newly founds ampacities
            s.args.I = df[VariableType.TRANSIT].values
            s.update()
            assert np.allclose(
                s.balance(
                    surface_temperature=df[TemperatureLocation.SURFACE],
                    core_temperature=df[TemperatureLocation.CORE],
                ).values,
                0.0,
                atol=tol,
            )
            assert np.allclose(
                s.morgan(
                    surface_temperature=df[TemperatureLocation.SURFACE],
                    core_temperature=df[TemperatureLocation.CORE],
                ).values,
                0.0,
                atol=tol,
            )
            # 3t solve
            dg = s.steady_temperature(
                surface_temperature_guess=df[TemperatureLocation.SURFACE]
                .round(1)
                .values,
                core_temperature_guess=df[TemperatureLocation.CORE].round(1).values,
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            # check consistency
            assert np.allclose(dg[temperature_at_location].values, 100.0)
