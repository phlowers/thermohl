# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone
import pytest
import numpy as np
import pandas as pd

from thermohl import power
from thermohl.solver.entities import PowerType, VariableType
from thermohl.solver.slv1t import Solver1T


@pytest.fixture
def solver():
    args = {
        "max_len": lambda: 1,
        "transit": np.array([0]),
        "ambient_temperature": np.array([25]),
        "wind_speed": np.array([0]),
        "wind_azimuth": np.array([0]),
        "ambient_pressure": np.array([101325]),
        "relative_humidity": np.array([50]),
        "precipitation_rate": np.array([0]),
        "linear_mass": 1.0,
        "heat_capacity": 1.0,
        "datetime_utc": datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
        "latitude": np.array([48.0]),
        "longitude": np.array([2.3]),
        "altitude": np.array([50.0]),
        "cable_azimuth": np.array([0.0]),
        "solar_absorptivity": np.array([0.5]),
    }
    cable = {
        "outer_diameter": np.array([3.186e-02]),
        "core_diameter": np.array([0.000]),
        "outer_area": np.array([6.004e-4]),
        "core_area": np.array([0.000]),
        "magnetic_coeff": np.array([1.000]),
        "magnetic_coeff_per_a": np.array([0.000]),
        "temperature_coeff_linear": np.array([3.600e-3]),
        "temperature_coeff_quadratic": np.array([8.000e-7]),
        "linear_resistance_dc_20c": np.array([5.540e-5]),
        "emissivity": np.array([0.8]),
    }
    args.update(cable)

    joule = power.rte.joule_heating.JouleHeating
    solar = power.rte.solar_heating.SolarHeating
    convective = power.rte.convective_cooling.ConvectiveCooling
    radiative = power.rte.radiative_cooling.RadiativeCooling

    solver = Solver1T(
        dic=args, joule=joule, solar=solar, convective=convective, radiative=radiative
    )
    return solver


def test_steady_temperature_default(solver):
    result = solver.steady_temperature()

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TEMPERATURE in result.columns
    assert PowerType.JOULE in result.columns
    assert PowerType.SOLAR in result.columns
    assert PowerType.CONVECTION in result.columns
    assert PowerType.RADIATION in result.columns
    assert PowerType.RAIN in result.columns


def test_steady_temperature_with_error(solver):
    result = solver.steady_temperature(return_err=True)

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TEMPERATURE in result.columns
    assert VariableType.ERROR in result.columns


def test_steady_temperature_no_power(solver):
    result = solver.steady_temperature(return_power=False)

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TEMPERATURE in result.columns
    assert PowerType.JOULE not in result.columns
    assert PowerType.SOLAR not in result.columns
    assert PowerType.CONVECTION not in result.columns
    assert PowerType.RADIATION not in result.columns
    assert PowerType.RAIN not in result.columns


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
    assert VariableType.TEMPERATURE in result.columns


def test_transient_temperature_default(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time)

    assert isinstance(result, dict)
    assert VariableType.TIME in result
    assert VariableType.TEMPERATURE in result
    assert len(result[VariableType.TIME]) == len(time)
    assert len(result[VariableType.TEMPERATURE]) == len(time)


def test_transient_temperature_with_initial_temp(solver):
    time = np.array([0, 1, 2, 3, 4, 5])
    T0 = 30.0

    result = solver.transient_temperature(time, T0=T0)

    assert isinstance(result, dict)
    assert VariableType.TIME in result
    assert VariableType.TEMPERATURE in result
    assert len(result[VariableType.TIME]) == len(time)
    assert len(result[VariableType.TEMPERATURE]) == len(time)
    assert result[VariableType.TEMPERATURE][0] == T0


def test_transient_temperature_with_error(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time, return_power=True)

    assert isinstance(result, dict)
    assert VariableType.TIME in result
    assert VariableType.TEMPERATURE in result
    assert len(result[VariableType.TIME]) == len(time)
    assert len(result[VariableType.TEMPERATURE]) == len(time)
    assert PowerType.JOULE in result
    assert PowerType.SOLAR in result
    assert PowerType.CONVECTION in result
    assert PowerType.RADIATION in result
    assert PowerType.RAIN in result


def test_steady_intensity_default(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature)

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TRANSIT in result.columns
    assert PowerType.JOULE in result.columns
    assert PowerType.SOLAR in result.columns
    assert PowerType.CONVECTION in result.columns
    assert PowerType.RADIATION in result.columns
    assert PowerType.RAIN in result.columns


def test_steady_intensity_with_error(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_err=True)

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TRANSIT in result.columns
    assert VariableType.ERROR in result.columns


def test_steady_intensity_no_power(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_power=False)

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TRANSIT in result.columns
    assert PowerType.JOULE not in result.columns
    assert PowerType.SOLAR not in result.columns
    assert PowerType.CONVECTION not in result.columns
    assert PowerType.RADIATION not in result.columns
    assert PowerType.RAIN not in result.columns


def test_steady_intensity_custom_params(solver):
    conductor_temperature = np.array([75])
    Imin = 5.0
    Imax = 1010.0
    tol = 1e-5
    maxiter = 100

    result = solver.steady_intensity(
        conductor_temperature, Imin=Imin, Imax=Imax, tol=tol, maxiter=maxiter
    )

    assert isinstance(result, pd.DataFrame)
    assert VariableType.TRANSIT in result.columns
