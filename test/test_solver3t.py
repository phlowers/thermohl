# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl import solver

_nprs = 123456


def _solvers(dic=None):
    return [
        solver._factory(dic=dic, heateq=he, model=m)
        for he in ["3t", "3tl"]
        for m in ["cner", "cigre", "ieee", "olla"]
    ]


def test_balance():
    # NB : this one fails only with 'cigre' powers

    tol = 1.0e-09
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        lat=np.random.uniform(42.0, 51.0, N),
        alt=np.random.uniform(0.0, 1600.0, N),
        azm=np.random.uniform(0.0, 360.0, N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        Ta=np.random.uniform(0.0, 30.0, N),
        ws=np.random.uniform(0.0, 7.0, N),
        wa=np.random.uniform(0.0, 90.0, N),
        I=np.random.uniform(40.0, 4000.0, N),
        d=np.random.randint(2, size=N) * solver.default_values()["d"],
    )

    for s in _solvers(dic):
        # compute guess with 1t solver
        s1 = solver._factory(dic=dic, heateq="1t", model="ieee")
        t1 = s1.steady_temperature(
            tol=2.0, maxiter=16, return_err=False, return_power=False
        )
        t1 = t1["t"].values
        # 3t solve
        df = s.steady_temperature(
            Tsg=t1, Tcg=t1, return_err=True, return_power=True, tol=tol, maxiter=64
        )
        # checks
        assert np.all(df["err"] < tol)
        assert np.allclose(
            s.balance(ts=df["t_surf"], tc=df["t_core"]).values, 0.0, atol=tol
        )
        assert np.allclose(
            s.morgan(ts=df["t_surf"], tc=df["t_core"]).values, 0.0, atol=tol
        )


def test_consistency():
    tol = 1.0e-09
    np.random.seed(_nprs)
    N = 9999
    dic = dict(
        lat=np.random.uniform(42.0, 51.0, N),
        alt=np.random.uniform(0.0, 1600.0, N),
        azm=np.random.uniform(0.0, 360.0, N),
        month=np.random.randint(1, 13, N),
        day=np.random.randint(1, 31, N),
        hour=np.random.randint(0, 24, N),
        Ta=np.random.uniform(0.0, 30.0, N),
        ws=np.random.uniform(0.0, 7.0, N),
        wa=np.random.uniform(0.0, 90.0, N),
        d=np.random.randint(2, size=N) * solver.default_values()["d"],
    )

    for s in _solvers(dic):
        for t in ["surf", "avg", "core"]:
            # solve intensity with different targets
            df = s.steady_intensity(
                T=100.0,
                target=t,
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            assert np.all(df["err"] < tol)
            # set args intensity to newly founds ampacities
            s.args.I = df["I"].values
            s.update()
            assert np.allclose(
                s.balance(ts=df["t_surf"], tc=df["t_core"]).values, 0.0, atol=tol
            )
            assert np.allclose(
                s.morgan(ts=df["t_surf"], tc=df["t_core"]).values, 0.0, atol=tol
            )
            # 3t solve
            dg = s.steady_temperature(
                Tsg=df["t_surf"].round(1).values,
                Tcg=df["t_core"].round(1).values,
                return_err=True,
                return_power=True,
                tol=1.0e-09,
                maxiter=64,
            )
            # check consistency
            assert np.allclose(dg["t_" + t].values, 100.0)
