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
        solver._factory(dic=dic, heateq="1t", model=m)
        for m in ["rte", "cigre", "ieee", "olla"]
    ]


def test_balance():
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
        transit=np.random.uniform(40.0, 4000.0, N),
        d=np.random.randint(2, size=N) * solver.default_values()["d"],
    )

    for s in _solvers(dic):
        dct = s.steady_temperature(
            return_err=True, return_power=True, tol=tol, maxiter=64
        )
        assert np.all(dct["err"] < tol)
        bl = np.abs(
            dct["P_joule"]
            + dct["P_solar"]
            - dct["P_convection"]
            - dct["P_radiation"]
            - dct["P_precipitation"]
        )
        atol = np.maximum(
            np.abs(s.balance(dct["t"] + 0.5 * dct["err"])),
            np.abs(s.balance(dct["t"] - 0.5 * dct["err"])),
        )
        assert np.allclose(bl, 0.0, atol=atol)


def test_consistency():
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
        result = s.steady_intensity(
            T=100.0, return_err=True, return_power=True, tol=1.0e-09, maxiter=64
        )
        bl = (
            result["P_joule"]
            + result["P_solar"]
            - result["P_convection"]
            - result["P_radiation"]
            - result["P_precipitation"]
        )
        assert np.allclose(bl, 0.0, atol=1.0e-06)
        s.args["transit"] = result["transit"]
        s.update()
        dg = s.steady_temperature(
            return_err=True, return_power=True, tol=1.0e-09, maxiter=64
        )
        assert np.allclose(dg["t"], 100.0)
