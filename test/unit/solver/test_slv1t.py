# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from thermohl.solver.slv1t import Solver1T


@pytest.fixture
def solver():
    args = {
        "max_len": lambda: 1,
        "I": np.array([0]),
        "Ta": np.array([25]),
        "ws": np.array([0]),
        "wa": np.array([0]),
        "Pa": np.array([101325]),
        "rh": np.array([50]),
        "pr": np.array([0]),
        "m": 1.0,
        "c": 1.0,
        "month": 1,
        "day": 1,
        "hour": 0,
    }
    return Solver1T(dic=args)


def test_steady_temperature_default(solver):
    result = solver.steady_temperature()

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.temp in result.columns
    assert Solver1T.Names.pjle in result.columns
    assert Solver1T.Names.psol in result.columns
    assert Solver1T.Names.pcnv in result.columns
    assert Solver1T.Names.prad in result.columns
    assert Solver1T.Names.ppre in result.columns


def test_steady_temperature_with_error(solver):
    result = solver.steady_temperature(return_err=True)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.temp in result.columns
    assert Solver1T.Names.err in result.columns


def test_steady_temperature_no_power(solver):
    result = solver.steady_temperature(return_power=False)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.temp in result.columns
    assert Solver1T.Names.pjle not in result.columns
    assert Solver1T.Names.psol not in result.columns
    assert Solver1T.Names.pcnv not in result.columns
    assert Solver1T.Names.prad not in result.columns
    assert Solver1T.Names.ppre not in result.columns


def test_steady_temperature_custom_params(solver):
    Tmin = 10.0
    Tmax = 50.0
    tol = 1e-5
    maxiter = 100

    result = solver.steady_temperature(
        Tmin=Tmin,
        Tmax=Tmax,
        tol=tol,
        maxiter=maxiter,
    )

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.temp in result.columns


def test_transient_temperature_default(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time)

    assert isinstance(result, dict)
    assert "time" in result
    assert "T" in result
    assert len(result["time"]) == len(time)
    assert len(result["T"]) == len(time)


def test_transient_temperature_with_initial_temp(solver):
    time = np.array([0, 1, 2, 3, 4, 5])
    T0 = 30.0

    result = solver.transient_temperature(time, T0=T0)

    assert isinstance(result, dict)
    assert "time" in result
    assert "T" in result
    assert len(result["time"]) == len(time)
    assert len(result["T"]) == len(time)
    assert result["T"][0] == T0


def test_transient_temperature_with_error(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time, return_power=True)

    assert isinstance(result, dict)
    assert "time" in result
    assert "T" in result
    assert len(result["time"]) == len(time)
    assert len(result["T"]) == len(time)
    assert Solver1T.Names.pjle in result
    assert Solver1T.Names.psol in result
    assert Solver1T.Names.pcnv in result
    assert Solver1T.Names.prad in result
    assert Solver1T.Names.ppre in result


def test_steady_intensity_default(solver):
    T = np.array([75])

    result = solver.steady_intensity(T)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.pjle in result.columns
    assert Solver1T.Names.psol in result.columns
    assert Solver1T.Names.pcnv in result.columns
    assert Solver1T.Names.prad in result.columns
    assert Solver1T.Names.ppre in result.columns


def test_steady_intensity_with_error(solver):
    T = np.array([75])

    result = solver.steady_intensity(T, return_err=True)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.err in result.columns


def test_steady_intensity_no_power(solver):
    T = np.array([75])

    result = solver.steady_intensity(T, return_power=False)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.pjle not in result.columns
    assert Solver1T.Names.psol not in result.columns
    assert Solver1T.Names.pcnv not in result.columns
    assert Solver1T.Names.prad not in result.columns
    assert Solver1T.Names.ppre not in result.columns


def test_steady_intensity_custom_params(solver):
    T = np.array([75])
    Imin = 5.0
    Imax = 100.0
    tol = 1e-5
    maxiter = 100

    result = solver.steady_intensity(T, Imin=Imin, Imax=Imax, tol=tol, maxiter=maxiter)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
