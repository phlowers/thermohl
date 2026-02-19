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
        "azm": array([90.0]),
        "ws": array([0]),
        "wa": array([45.0]),
        "alt": array([100]),
        "Ta": array([20]),
        "transit": array([1500.0]),
        "I0": array([500]),
        "error_code": array([0]),
        "t_core": array([36.164476]),
        "t_surf": array([34.955032]),
        "t_avg": array([35.559754]),
        "temp_error": array([0]),
        "buffer_start": array([1.6150752e09]),
        "D": array([0.03105]),
        "d": array([0.0]),
        "A": array([0.00057]),
        "a": array([0.0]),
        "RDC20": array([5.83e-05]),
        "m": array([1.539]),
        "c": array([900.0]),
        "R": array([0]),
        "l": array([1]),
        "kl": array([0.0036]),
        "kq": array([8.0e-07]),
        "km": array([1.0]),
        "ki": array([0.0]),
        "alpha": array([0.9]),
        "epsilon": array([0.8]),
        "month": array([3], dtype=int32),
        "day": array([7], dtype=int32),
        "hour": array([0.0]),
    }
    Ts0 = [34.955032]
    Tc0 = [36.164476]

    solver = rte(data, heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY)
    result = solver.transient_temperature_legacy(
        time=np.linspace(0, 60, 61),
        Ts0=Ts0,
        Tc0=Tc0,
        return_power=True,
    )

    print(result)
    assert abs(result[TemperatureLocation.CORE][-1] - 42) <= 0.5
    assert abs(result[TemperatureLocation.SURFACE][-1] - 39.9) <= 0.5
    assert abs(result[TemperatureLocation.AVERAGE][-1] - 40.9) <= 0.5
