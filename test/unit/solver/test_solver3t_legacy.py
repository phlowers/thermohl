# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone
import pytest
import numpy as np
from numpy import array
from thermohl.solver.entities import TemperatureType, HeatEquationType

from thermohl.solver import rte
from thermohl.solver.solver import temporarily_override_parameter


def test_solver3t_legacy():
    data = {
        "cable_azimuth": array([90.0]),
        "wind_speed": array([0]),
        "wind_azimuth": array([45.0]),
        "altitude": array([100]),
        "ambient_temperature": array([20]),
        "transit": array([1500.0]),
        "outer_diameter": array([0.03105]),
        "core_diameter": array([0.0]),
        "outer_area": array([0.00057]),
        "core_area": array([0.0]),
        "linear_resistance_dc_20c": array([5.83e-05]),
        "linear_mass": array([1.539]),
        "heat_capacity": array([900.0]),
        "roughness_ratio": array([0]),
        "radial_thermal_conductivity": array([1]),
        "temperature_coeff_linear": array([0.0036]),
        "temperature_coeff_quadratic": array([8.0e-07]),
        "magnetic_coeff": array([1.0]),
        "magnetic_coeff_per_a": array([0.0]),
        "solar_absorptivity": array([0.9]),
        "emissivity": array([0.8]),
        "datetime_utc": datetime(2000, 3, 7, 0, tzinfo=timezone.utc),
    }
    Ts0 = [34.955032]
    Tc0 = [36.164476]

    solver = rte(data, heat_equation=HeatEquationType.THREE_TEMPERATURES_LEGACY)
    result = solver.transient_temperature_legacy(
        offset=np.linspace(0, 60, 61),
        surface_temperature_0=Ts0,
        core_temperature_0=Tc0,
        return_power=True,
    )

    print(result)
    assert abs(result[TemperatureType.CORE.value][-1] - 42) <= 0.5
    assert abs(result[TemperatureType.SURFACE.value][-1] - 39.9) <= 0.5
    assert abs(result[TemperatureType.AVERAGE.value][-1] - 40.9) <= 0.5


def test_steady_temperature_uncertainty():
    solver_input_data = {
        "cable_azimuth": array([90.0, 85.0]),
        "wind_speed": array([0, 1.0]),
        "wind_azimuth": array([45.0, 43.0]),
        "altitude": array([100]),
        "ambient_temperature": array([20.0, 22.0]),
        "transit": array([1500.0, 1550.0]),
        "outer_diameter": array([0.03105]),
        "core_diameter": array([0.0]),
        "outer_area": array([0.00057]),
        "core_area": array([0.0]),
        "linear_resistance_dc_20c": array([5.83e-05]),
        "linear_mass": array([1.539]),
        "heat_capacity": array([900.0]),
        "roughness_ratio": array([0]),
        "radial_thermal_conductivity": array([1]),
        "temperature_coeff_linear": array([0.0036]),
        "temperature_coeff_quadratic": array([8.0e-07]),
        "magnetic_coeff": array([1.0]),
        "magnetic_coeff_per_a": array([0.0]),
        "solar_absorptivity": array([0.9]),
        "emissivity": array([0.8]),
        "datetime_utc": datetime(2026, 12, 13, 13, 12, tzinfo=timezone.utc),
    }
    solver = rte(
        solver_input_data, heat_equation=HeatEquationType.THREE_TEMPERATURES_LEGACY
    )

    saved_transit = solver.args.transit.copy()
    saved_ambient_temperature = solver.args.ambient_temperature.copy()
    saved_wind_speed = solver.args.wind_speed.copy()
    saved_wind_azimuth = solver.args.wind_azimuth.copy()

    saved_solar_irradiance = solver.solar_heating.solar_irradiance.copy()
    saved_solar_heating = solver.solar_heating.value(100).copy()

    result = solver.steady_temperature(return_uncertainty=True)

    assert len(result["uncertainty"]) == len(solver_input_data["transit"])

    # check that solver args haven't been changed
    assert np.allclose(solver.args.transit, saved_transit)
    assert np.allclose(solver.args.ambient_temperature, saved_ambient_temperature)
    assert np.allclose(solver.args.wind_speed, saved_wind_speed)
    assert np.allclose(solver.args.wind_azimuth, saved_wind_azimuth)
    assert np.allclose(solver.solar_heating.solar_irradiance, saved_solar_irradiance)
    assert np.allclose(solver.solar_heating.value(100), saved_solar_heating)


def test_temporarily_override_parameter():
    solver = rte({}, heat_equation=HeatEquationType.THREE_TEMPERATURES_LEGACY)
    with pytest.raises(ValueError):
        with temporarily_override_parameter(solver, "made_up_parameter", 42):
            pass  # noqa
