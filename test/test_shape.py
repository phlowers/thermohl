# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from thermohl import solver
from thermohl.solver import HeatEquationType, SolverType
from thermohl.solver.enums.cable_location import CableLocation
from thermohl.solver.enums.temperature_location import TemperatureLocation
from thermohl.solver.enums.variable_type import VariableType


def _solvers():
    li = []
    for heat_equation in [
        HeatEquationType.WITH_ONE_TEMPERATURE,
        HeatEquationType.WITH_THREE_TEMPERATURES,
    ]:
        for m in [
            SolverType.SOLVER_RTE,
            SolverType.SOLVER_CIGRE,
            SolverType.SOLVER_IEEE,
            SolverType.SOLVER_OLLA,
        ]:
            li.append(solver._factory(dic=None, heat_equation=heat_equation, model=m))
    return li


def _ampargs(s: solver.Solver, t: pd.DataFrame):
    if isinstance(s, solver.Solver1T):
        a = dict(max_conductor_temperature=t[VariableType.TEMPERATURE].values)
    elif isinstance(s, solver.Solver3T):
        a = dict(
            max_conductor_temperature=t[TemperatureLocation.SURFACE].values,
            target=CableLocation.SURFACE,
        )
    else:
        raise NotImplementedError
    return a


def _traargs(s: solver.Solver, ds: pd.DataFrame, t):
    if isinstance(s, solver.Solver1T):
        a = dict(time=t, T0=ds[VariableType.TEMPERATURE].values)
    elif isinstance(s, solver.Solver3T):
        a = dict(
            time=t,
            surface_temperature_0=ds[TemperatureLocation.SURFACE].values,
            core_temperature_0=ds[TemperatureLocation.CORE].values,
        )
    else:
        raise NotImplementedError
    return a


def test_power_default():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    for s in _solvers():
        for p in [
            s.joule_heating,
            s.solar_heating,
            s.convective_cooling,
            s.radiative_cooling,
            s.precipitation_cooling,
        ]:
            p.__init__(**s.args.__dict__)
            assert np.isscalar(p.value(0.0))
            assert p.value(np.array([0.0])).shape == (1,)
            assert p.value(np.array([0.0, 10.0])).shape == (2,)


def test_power_1d():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    n = 61
    for s in _solvers():
        d = s.args.__dict__.copy()
        d[VariableType.TRANSIT.value] = np.linspace(0.0, +999.0, n)
        d["solar_absorptivity"] = np.linspace(0.5, 0.9, n)
        d["ambient_temperature"] = np.linspace(-10.0, +50.0, n)
        for p in [
            s.joule_heating,
            s.solar_heating,
            s.convective_cooling,
            s.radiative_cooling,
            s.precipitation_cooling,
        ]:
            p.__init__(**d)
            v = p.value(0.0)
            assert np.isscalar(v) or v.shape == (n,)
            v = p.value(np.array([0.0]))
            assert v.shape == (1,) or v.shape == (n,)
            assert p.value(np.linspace(-10, +50, n)).shape == (n,)


def test_steady_default():
    for s in _solvers():
        t = s.steady_temperature()
        a = _ampargs(s, t)
        i = s.steady_intensity(**a)
        assert len(t) == 1
        assert len(i) == 1


def test_steady_1d():
    n = 61
    for s in _solvers():
        s.args.ambient_temperature = np.linspace(-10, +50, n)
        s.update()
        t = s.steady_temperature()
        a = _ampargs(s, t)
        i = s.steady_intensity(**a)
        assert len(t) == n
        assert len(i) == n


def test_steady_1d_mix():
    n = 61
    for s in _solvers():
        s.args.ambient_temperature = np.linspace(-10, +50, n)
        s.args.transit = np.array([199.0])
        s.update()
        t = s.steady_temperature()
        a = _ampargs(s, t)
        i = s.steady_intensity(**a)
        assert len(t) == n
        assert len(i) == n


def test_transient_0():
    for s in _solvers():
        t = np.linspace(0, 3600, 361)

        ds = s.steady_temperature()
        a = _traargs(s, ds, t)

        r = s.transient_temperature(**a)
        assert len(r.pop(VariableType.TIME)) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t),)

        r = s.transient_temperature(**{**a, "return_power": True})

        assert len(r.pop(VariableType.TIME)) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t),)


def test_transient_1():
    n = 7
    for s in _solvers():
        s.args.ambient_temperature = np.linspace(-10, +50, n)
        s.update()

        t = np.linspace(0, 3600, 361)

        ds = s.steady_temperature()
        a = _traargs(s, ds, t)

        r = s.transient_temperature(**a)
        assert len(r.pop(VariableType.TIME)) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t), n)

        r = s.transient_temperature(**{**a, "return_power": True})
        assert len(r.pop(VariableType.TIME)) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t), n)
