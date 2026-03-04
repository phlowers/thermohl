# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone

from zoneinfo import ZoneInfo
import os.path
import numpy as np
import pandas as pd

from functional_test.functional_test_steady import get_scenarios, get_cable_data
from thermohl.solver import rte, HeatEquationType, TemperatureLocation
from thermohl.solver.enums.variable_type import VariableType


# todo : faire une fonction scn2dict pour le transient
def scn2dict(scn: dict) -> dict:
    """Convert scenario to thermohl input."""
    dic = get_cable_data(scn["Conducteur"])

    dic["latitude"] = scn["Lat"]
    dic["longitude"] = scn["Lon"]
    dic["altitude"] = scn["Altitude"]
    dic["cable_azimuth"] = scn["Azimut_portee"]
    dic["ambient_temperature"] = scn["T_amb"]
    dic["wind_speed"] = scn["V"]
    dic["wind_azimuth"] = scn["Azimut_V"]
    dic["albedo"] = scn["Albedo"]
    dic["measured_solar_irradiance"] = scn["QG_mesure"]
    dic["nebulosity"] = scn["Neb"]
    dic["solar_absorptivity"] = 0.9
    dic["emissivity"] = 0.8

    datetime_paris = datetime.strptime(
        f'{scn["Date"]} {scn["Heure"]}', "%d/%m/%Y %H:%M"
    ).replace(tzinfo=ZoneInfo("Europe/Paris"))
    datetime_utc = datetime_paris.astimezone(timezone.utc)
    dic["datetime_utc"] = datetime_utc

    return dic


def test_transient_temperature():
    time_constant = 600.0
    offsets = np.array([60, 120])  # np.array([1, 2, 3, 5, 10, 30]) * 60

    i = 0
    for scenario in get_scenarios("scenarios_transient.csv"):
        i += 1
        print(f"YOMAN num {i}: {scenario=}")

        solver = rte(
            scn2dict(scenario),
            heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
        )

        # initial steady state
        solver.args[VariableType.TRANSIT.value] = scenario["I_initial"]
        solver.update()
        initial_state = solver.steady_temperature()

        # final steady state
        solver.args[VariableType.TRANSIT.value] = scenario["I_final"]
        solver.update()
        final_state = solver.steady_temperature(
            surface_temperature_guess=scenario["T_surf_final"],
            core_temperature_guess=scenario["T_heart_final"],
        )

        # transient temperature (linearized)
        transient_result = solver.transient_temperature_legacy(
            offset=offsets,
            surface_temperature_0=initial_state[TemperatureLocation.SURFACE],
            core_temperature_0=initial_state[TemperatureLocation.CORE],
            time_constant=time_constant,
        )

        # print(
        #     f'TEST : {scenario["T_mean_final"]=}, {final_state[TemperatureLocation.AVERAGE][0]=}'
        # )
        # print(f"TEST2 : {final_state[TemperatureLocation.AVERAGE]=}")
        # todo : ameliorer la tolerance
        # check final temperature
        assert np.isclose(
            scenario["T_mean_final"],
            final_state[TemperatureLocation.AVERAGE][0],
            atol=3,
        )

        print(f"\n\nYOMAN2 : {transient_result=}")
        offsets_minute = offsets // 60
        expected_surface_temperatures = [
            scenario[f"T_surf_{offset}"] for offset in offsets_minute
        ]
        expected_mean_temperatures = [
            scenario[f"T_mean_{offset}"] for offset in offsets_minute
        ]
        expected_core_temperatures = [
            scenario[f"T_heart_{offset}"] for offset in offsets_minute
        ]

        assert np.allclose(
            transient_result[TemperatureLocation.SURFACE], expected_surface_temperatures
        )
        assert np.allclose(
            transient_result[TemperatureLocation.AVERAGE], expected_mean_temperatures
        )
        assert np.allclose(
            transient_result[TemperatureLocation.CORE], expected_core_temperatures
        )
