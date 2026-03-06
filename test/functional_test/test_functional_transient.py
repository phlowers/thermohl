# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from functional_test.test_functional_steady import (
    get_scenarios,
    scn2dict,
)
from thermohl.solver import rte, HeatEquationType, TemperatureLocation
from thermohl.solver.enums.variable_type import VariableType


def test_transient_temperature():
    time_constant = 600.0
    offsets = np.arange(0.0, 1800.0 + 1, 10)

    for scenario in get_scenarios("scenarios_transient.csv"):
        solver = rte(
            scn2dict(scenario),
            heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
        )

        # initial steady state
        solver.args[VariableType.TRANSIT.value] = scenario["I_initial"]
        solver.update()
        initial_state = solver.steady_temperature()

        # check initial temperature
        assert np.isclose(
            scenario["T_mean_0"],
            initial_state[TemperatureLocation.AVERAGE][0],
            atol=0.01,
        )

        # final steady state
        solver.args[VariableType.TRANSIT.value] = scenario["I_final"]
        solver.update()
        final_state = solver.steady_temperature(
            surface_temperature_guess=scenario["T_surf_final"],
            core_temperature_guess=scenario["T_heart_final"],
        )

        # check final temperature
        assert np.isclose(
            scenario["T_mean_final"],
            final_state[TemperatureLocation.AVERAGE][0],
            atol=0.05,
        )

        # transient temperature
        transient_result = solver.transient_temperature_legacy(
            offset=offsets,
            surface_temperature_0=initial_state[TemperatureLocation.SURFACE],
            core_temperature_0=initial_state[TemperatureLocation.CORE],
            time_constant=time_constant,
        )

        # get expected temperatures from scenario at the right offsets
        useful_offsets = np.array([0, 1, 2, 3, 5, 10, 30])  # in minutes
        expected_surface_temperatures = [
            scenario[f"T_surf_{offset}"] for offset in useful_offsets
        ]
        expected_mean_temperatures = [
            scenario[f"T_mean_{offset}"] for offset in useful_offsets
        ]
        expected_core_temperatures = [
            scenario[f"T_heart_{offset}"] for offset in useful_offsets
        ]

        atol = 0.01
        # get indexes of the useful offsets in the thermohl computation of transient temperature.
        useful_indexes = np.searchsorted(
            transient_result[VariableType.TIME], useful_offsets * 60
        )
        assert np.allclose(
            transient_result[TemperatureLocation.SURFACE][useful_indexes],
            expected_surface_temperatures,
            atol=atol,
        )
        assert np.allclose(
            transient_result[TemperatureLocation.AVERAGE][useful_indexes],
            expected_mean_temperatures,
            atol=atol,
        )
        assert np.allclose(
            transient_result[TemperatureLocation.CORE][useful_indexes],
            expected_core_temperatures,
            atol=atol,
        )
