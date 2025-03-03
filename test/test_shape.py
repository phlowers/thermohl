# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from thermohl import solver


def _solvers():
    li = []
    for ht in ["1t", "3t"]:
        for m in ["cner", "cigre", "ieee", "olla"]:
            li.append(solver._factory(dic=None, heateq=ht, model=m))
    return li


def _ampargs(s: solver.Solver, t: pd.DataFrame):
    if isinstance(s, solver.Solver1T):
        a = dict(T=t[solver.Solver.Names.temp].values)
    elif isinstance(s, solver.Solver3T):
        a = dict(T=t[solver.Solver.Names.tsurf].values, target=solver.Solver.Names.surf)
    else:
        raise NotImplementedError
    return a


def _traargs(s: solver.Solver, ds: pd.DataFrame, t, I):
    if isinstance(s, solver.Solver1T):
        a = dict(time=t, T0=ds[solver.Solver.Names.temp].values)
    elif isinstance(s, solver.Solver3T):
        a = dict(
            time=t,
            Ts0=ds[solver.Solver.Names.tsurf].values,
            Tc0=ds[solver.Solver.Names.tcore].values,
        )
    else:
        raise NotImplementedError
    return a


def test_power_default():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    for s in _solvers():
        for p in [s.jh, s.sh, s.cc, s.rc, s.pc]:
            p.__init__(**s.args.__dict__)
            assert np.isscalar(p.value(0.0))
            assert p.value(np.array([0.0])).shape == (1,)
            assert p.value(np.array([0.0, 10.0])).shape == (2,)


def test_power_1d():
    """Check that PowerTerm.value(x) returns correct shape depending on init dict and temperature input."""
    n = 61
    for s in _solvers():
        d = s.args.__dict__.copy()
        d["I"] = np.linspace(0.0, +999.0, n)
        d["alpha"] = np.linspace(0.5, 0.9, n)
        d["Ta"] = np.linspace(-10.0, +50.0, n)
        for p in [s.jh, s.sh, s.cc, s.rc, s.pc]:
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
        s.args.Ta = np.linspace(-10, +50, n)
        s.update()
        t = s.steady_temperature()
        a = _ampargs(s, t)
        i = s.steady_intensity(**a)
        assert len(t) == n
        assert len(i) == n


def test_steady_1d_mix():
    n = 61
    for s in _solvers():
        s.args.Ta = np.linspace(-10, +50, n)
        s.args.I = np.array([199.0])
        s.update()
        t = s.steady_temperature()
        a = _ampargs(s, t)
        i = s.steady_intensity(**a)
        assert len(t) == n
        assert len(i) == n


def test_transient_0():
    for s in _solvers():
        t = np.linspace(0, 3600, 361)
        I = 199 * np.ones_like(t)

        ds = s.steady_temperature()
        a = _traargs(s, ds, t, I)

        r = s.transient_temperature(**a)
        assert len(r.pop("time")) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t),)

        r = s.transient_temperature(**{**a, "return_power": True})

        assert len(r.pop("time")) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t),)


def test_transient_1():
    n = 7
    for s in _solvers():
        s.args.Ta = np.linspace(-10, +50, n)
        s.update()

        t = np.linspace(0, 3600, 361)
        I = 199 * np.ones_like(t)

        ds = s.steady_temperature()
        a = _traargs(s, ds, t, I)

        r = s.transient_temperature(**a)
        assert len(r.pop("time")) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t), n)

        r = s.transient_temperature(**{**a, "return_power": True})
        assert len(r.pop("time")) == len(t)
        for k in r.keys():
            assert r[k].shape == (len(t), n)
