# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np

from thermohl import power
from thermohl.solver.entities import PowerType, VariableType
from thermohl.solver.slv1t import Solver1T
from test.utils import get_cable_data


@pytest.fixture
def solver_args():
    args = {
        "max_len": lambda: 1,
        "transit": np.array([0]),
        "ambient_temperature": np.array([25]),
        "wind_speed": np.array([0]),
        "wind_azimuth": np.array([0]),
        "ambient_pressure": np.array([101325]),
        "relative_humidity": np.array([50]),
        "precipitation_rate": np.array([0]),
        "datetime_utc": np.datetime64("2025-01-01T00:00:00"),
        "latitude": np.array([48.0]),
        "longitude": np.array([2.3]),
        "altitude": np.array([50.0]),
        "cable_azimuth": np.array([0.0]),
        "solar_absorptivity": np.array([0.5]),
    }
    cable = {
        "linear_mass": 1.0,
        "heat_capacity": 1.0,
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
    return args


@pytest.fixture
def solver(solver_args):
    return create_solver(solver_args)


def create_solver(solver_args):
    joule = power.rte.joule_heating.JouleHeating
    solar = power.rte.solar_heating.SolarHeating
    convective = power.rte.convective_cooling.ConvectiveCooling
    radiative = power.rte.radiative_cooling.RadiativeCooling

    solver = Solver1T(
        dic=solver_args,
        joule=joule,
        solar=solar,
        convective=convective,
        radiative=radiative,
    )
    return solver


def test_steady_temperature_default(solver):
    result = solver.steady_temperature()

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TEMPERATURE.value in result
    assert PowerType.JOULE.value in result
    assert PowerType.SOLAR.value in result
    assert PowerType.CONVECTION.value in result
    assert PowerType.RADIATION.value in result
    assert PowerType.RAIN.value in result


def test_steady_temperature_with_error(solver):
    result = solver.steady_temperature(return_err=True)

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TEMPERATURE.value in result
    assert VariableType.ERROR.value in result


def test_steady_temperature_no_power(solver):
    result = solver.steady_temperature(return_power=False)

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TEMPERATURE.value in result
    assert PowerType.JOULE.value not in result
    assert PowerType.SOLAR.value not in result
    assert PowerType.CONVECTION.value not in result
    assert PowerType.RADIATION.value not in result
    assert PowerType.RAIN.value not in result


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

    assert isinstance(result, dict)
    for key in result.keys():
        assert isinstance(result[key], np.ndarray)
    assert VariableType.TEMPERATURE.value in result


def test_transient_temperature_default(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time)

    assert isinstance(result, dict)
    assert VariableType.TIME.value in result
    assert VariableType.TEMPERATURE.value in result
    assert len(result[VariableType.TIME.value]) == len(time)
    assert len(result[VariableType.TEMPERATURE.value]) == len(time)


def test_transient_temperature_with_initial_temp(solver):
    time = np.array([0, 1, 2, 3, 4, 5])
    T0 = 30.0

    result = solver.transient_temperature(time, T0=T0)

    assert isinstance(result, dict)
    assert VariableType.TIME.value in result
    assert VariableType.TEMPERATURE.value in result
    assert len(result[VariableType.TIME.value]) == len(time)
    assert len(result[VariableType.TEMPERATURE.value]) == len(time)
    assert result[VariableType.TEMPERATURE.value][0] == T0


def test_transient_temperature_with_error(solver):
    time = np.array([0, 1, 2, 3, 4, 5])

    result = solver.transient_temperature(time, return_power=True)

    assert isinstance(result, dict)
    assert VariableType.TIME.value in result
    assert VariableType.TEMPERATURE.value in result
    assert len(result[VariableType.TIME.value]) == len(time)
    assert len(result[VariableType.TEMPERATURE.value]) == len(time)
    assert PowerType.JOULE.value in result
    assert PowerType.SOLAR.value in result
    assert PowerType.CONVECTION.value in result
    assert PowerType.RADIATION.value in result
    assert PowerType.RAIN.value in result


def test_steady_intensity_default(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature)

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TRANSIT.value in result
    assert PowerType.JOULE.value in result
    assert PowerType.SOLAR.value in result
    assert PowerType.CONVECTION.value in result
    assert PowerType.RADIATION.value in result
    assert PowerType.RAIN.value in result


def test_steady_intensity_with_error(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_err=True)

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TRANSIT.value in result
    assert VariableType.ERROR.value in result


def test_steady_intensity_no_power(solver):
    conductor_temperature = np.array([75])

    result = solver.steady_intensity(conductor_temperature, return_power=False)

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TRANSIT.value in result
    assert PowerType.JOULE.value not in result
    assert PowerType.SOLAR.value not in result
    assert PowerType.CONVECTION.value not in result
    assert PowerType.RADIATION.value not in result
    assert PowerType.RAIN.value not in result


def test_steady_intensity_custom_params(solver):
    conductor_temperature = np.array([75])
    Imin = 5.0
    Imax = 1010.0
    tol = 1e-5
    maxiter = 100

    result = solver.steady_intensity(
        conductor_temperature, Imin=Imin, Imax=Imax, tol=tol, maxiter=maxiter
    )

    assert isinstance(result, dict)
    for value in result.values():
        assert isinstance(value, np.ndarray)
    assert VariableType.TRANSIT.value in result


def test_reduced_intensity_doesn_t_change_solver_args() -> None:
    args = {}
    solver = create_solver(args)

    saved_ambient_temperature = solver.args.ambient_temperature
    saved_wind_speed = solver.args.wind_speed
    saved_solver_wind_attack_angle = solver.args.wind_attack_angle
    saved_wind_attack_angle = solver.convective_cooling.wind_attack_angle
    saved_transit = solver.args.transit
    saved_solar_heating = solver.solar_heating

    solver.reduced_intensity(
        measured_temperature_difference=10.0,
        measured_intensity=360.0,
        ambient_temperature=25.0,
        wind_speed=4.0,
        solar_irradiance=800.0,
        max_conductor_temperature=120.0,
    )

    # Check that solver args and power term attributes have not been changed
    np.testing.assert_equal(solver.args.ambient_temperature, saved_ambient_temperature)
    np.testing.assert_equal(solver.args.wind_speed, saved_wind_speed)
    np.testing.assert_equal(solver.args.transit, saved_transit)
    np.testing.assert_equal(solver.joule_heating.transit, saved_transit)
    np.testing.assert_equal(
        solver.args.wind_attack_angle,
        saved_solver_wind_attack_angle,
    )
    np.testing.assert_equal(
        solver.args.wind_attack_angle,
        saved_solver_wind_attack_angle,
    )
    np.testing.assert_equal(
        solver.convective_cooling.wind_attack_angle,
        saved_wind_attack_angle,
    )
    assert solver.solar_heating == saved_solar_heating


@pytest.mark.parametrize(
    "measured_temperature_difference, expected_result",
    [
        (0.1, 980),
        (10, 604),
        (20, 470),
        (50, 318),
    ],
    ids=[
        "delta T = 0.1",
        "delta T = 10",
        "delta T = 20",
        "delta T = 50",
    ],
)
def test_reduced_intensity_crocus400__varying_temperature_difference(
    measured_temperature_difference, expected_result
) -> None:
    cable = get_cable_data("CROCUS400")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )
    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=measured_temperature_difference,
        measured_intensity=300,
        ambient_temperature=30.0,
        wind_speed=0.6,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, expected_result, atol=1)


@pytest.mark.parametrize(
    "measured_intensity, expected_result, atol",
    [
        (500, 638, 1),
        (600, 733, 2),
    ],
    ids=[
        "500 A",
        "600 A",
    ],
)
def test_reduced_intensity_aster600__varying_measured_intensity(
    measured_intensity,
    expected_result,
    atol,
) -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=30,
        measured_intensity=measured_intensity,
        ambient_temperature=30.0,
        wind_speed=0.6,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, expected_result, atol=atol)


@pytest.mark.parametrize(
    "measured_intensity, expected_result",
    [
        (500, 589),
        (600, 659),
    ],
    ids=[
        "500 A",
        "600 A",
    ],
)
def test_reduced_intensity_crocus400__varying_measured_intensity(
    measured_intensity, expected_result
) -> None:
    cable = get_cable_data("CROCUS400")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=30,
        measured_intensity=measured_intensity,
        ambient_temperature=30.0,
        wind_speed=0.6,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, expected_result, atol=1)


@pytest.mark.parametrize(
    "measured_temperature_difference, expected_result, atol",
    [
        (0.1, 1342, 10),
        (1, 1188, 10),
        (5, 841, 2),
        (10, 658, 2),
        (20, 493, 1),
        (50, 323, 1),
    ],
    ids=[
        "delta T = 0.1",
        "delta T = 1",
        "delta T = 5",
        "delta T = 10",
        "delta T = 20",
        "delta T = 50",
    ],
)
def test_reduced_intensity_aster600__varying_temperature_difference(
    measured_temperature_difference,
    expected_result,
    atol,
) -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=measured_temperature_difference,
        measured_intensity=300,
        ambient_temperature=30.0,
        wind_speed=0.6,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, expected_result, atol=atol)


def test_reduced_intensity_aster600__other_ambient_temperature() -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=20,
        measured_intensity=450,
        ambient_temperature=10,
        wind_speed=0.6,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, 791, atol=2)


def test_reduced_intensity_aster600__other_wind_speed() -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=20,
        measured_intensity=450,
        ambient_temperature=10,
        wind_speed=0.001,
        solar_irradiance=600.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, 717, atol=2)


def test_reduced_intensity_aster600__other_solar_irradiance() -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=20,
        measured_intensity=450,
        ambient_temperature=10,
        wind_speed=0.6,
        solar_irradiance=100.0,
        max_conductor_temperature=100.0,
    )

    np.testing.assert_allclose(result, 828, atol=2)


def test_reduced_intensity_aster600__other_max_conductor_temperature() -> None:
    cable = get_cable_data("ASTER600")
    cable.update(
        {
            "solar_absorptivity": 0.9,
            "emissivity": 0.8,
        }
    )

    solver = create_solver(cable)

    result = solver.reduced_intensity(
        measured_temperature_difference=20,
        measured_intensity=450,
        ambient_temperature=10,
        wind_speed=0.6,
        solar_irradiance=100.0,
        max_conductor_temperature=75,
    )

    np.testing.assert_allclose(result, 702, atol=2)
