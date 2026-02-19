# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np
import pandas as pd
from thermohl.solver.slv1t import Solver1T


@pytest.fixture
def solver():
    args = {
        "max_len": lambda: 1,
        "transit": np.array([0]),
        "ambient_temperature": np.array([25]),
        "wind_speed": np.array([0]),
        "wind_angle": np.array([0]),
        "ambient_pressure": np.array([101325]),
        "relative_humidity": np.array([50]),
        "precipitation_rate": np.array([0]),
        "linear_mass": 1.0,
        "heat_capacity": 1.0,
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
    assert "conductor_temperature" in result
    assert len(result["time"]) == len(time)
    assert len(result["conductor_temperature"]) == len(time)


def test_transient_temperature_with_initial_temp(solver):
    time = np.array([0, 1, 2, 3, 4, 5])
    T0 = 30.0

    result = solver.transient_temperature(time, T0=T0)

    assert isinstance(result, dict)
    assert "time" in result
    assert "conductor_temperature" in result
    assert len(result["time"]) == len(time)
    assert len(result["conductor_temperature"]) == len(time)
    assert result["conductor_temperature"][0] == T0


def test_transient_temperature_with_error(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time, return_power=True)

    assert isinstance(result, dict)
    assert "time" in result
    assert "conductor_temperature" in result
    assert len(result["time"]) == len(time)
    assert len(result["conductor_temperature"]) == len(time)
    assert Solver1T.Names.pjle in result
    assert Solver1T.Names.psol in result
    assert Solver1T.Names.pcnv in result
    assert Solver1T.Names.prad in result
    assert Solver1T.Names.ppre in result


def test_steady_intensity_default(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.pjle in result.columns
    assert Solver1T.Names.psol in result.columns
    assert Solver1T.Names.pcnv in result.columns
    assert Solver1T.Names.prad in result.columns
    assert Solver1T.Names.ppre in result.columns


def test_steady_intensity_with_error(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_err=True)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.err in result.columns


def test_steady_intensity_no_power(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_power=False)

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
    assert Solver1T.Names.pjle not in result.columns
    assert Solver1T.Names.psol not in result.columns
    assert Solver1T.Names.pcnv not in result.columns
    assert Solver1T.Names.prad not in result.columns
    assert Solver1T.Names.ppre not in result.columns


def test_steady_intensity_custom_params(solver):
    conductor_temperature = np.array([75])
    Imin = 5.0
    Imax = 100.0
    tol = 1e-5
    maxiter = 100

    result = solver.steady_intensity(
        conductor_temperature, Imin=Imin, Imax=Imax, tol=tol, maxiter=maxiter
    )

    assert isinstance(result, pd.DataFrame)
    assert Solver1T.Names.transit in result.columns
