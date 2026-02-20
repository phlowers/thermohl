# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from numpy import array, int32
from thermohl.solver.enums.temperature_location import TemperatureLocation

from thermohl.solver.enums.heat_equation_type import HeatEquationType

from thermohl.solver import rte


def test_solver3t_legacy():
    data = {
        "cpo_nbr_cable": array([1]),
        "azimuth": array([90.0]),
        "wind_speed": array([0]),
        "wind_angle": array([45.0]),
        "altitude": array([100]),
        "ambient_temperature": array([20]),
        "transit": array([1500.0]),
        "I0": array([500]),
        "error_code": array([0]),
        "t_core": array([36.164476]),
        "t_surf": array([34.955032]),
        "t_avg": array([35.559754]),
        "temp_error": array([0]),
        "buffer_start": array([1.6150752e09]),
        "outer_diameter": array([0.03105]),
        "core_diameter": array([0.0]),
        "outer_area": array([0.00057]),
        "core_area": array([0.0]),
        "linear_resistance_dc_20c": array([5.83e-05]),
        "m": array([1.539]),
        "heat_capacity": array([900.0]),
        "roughness_ratio": array([0]),
        "radial_thermal_conductivity": array([1]),  # FIXME?
        "temperature_coeff_linear": array([0.0036]),
        "temperature_coeff_quadratic": array([8.0e-07]),
        "magnetic_coeff": array([1.0]),
        "magnetic_coeff_per_a": array([0.0]),
        "solar_absorptivity": array([0.9]),
        "emissivity": array([0.8]),
        "month": array([3], dtype=int32),
        "day": array([7], dtype=int32),
        "hour": array([0.0]),
    }
    Ts0 = [34.955032]
    Tc0 = [36.164476]

    solver = rte(data, heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY)
    result = solver.transient_temperature_legacy(
        time=np.linspace(0, 60, 61),
        surface_temperature_0=Ts0,
        core_temperature_0=Tc0,
        return_power=True,
    )

    print(result)
    assert abs(result[TemperatureLocation.CORE][-1] - 42) <= 0.5
    assert abs(result[TemperatureLocation.SURFACE][-1] - 39.9) <= 0.5
    assert abs(result[TemperatureLocation.AVERAGE][-1] - 40.9) <= 0.5
