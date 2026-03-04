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
    # utile uniquement pour le test test_steady_temperature
    dic[VariableType.TRANSIT.value] = scn["Intensite"]

    return dic


# def test_transient_temperature():
#     time_constant = 600.0
#     offsets = [1, 2, 3, 4, 5, 10, 30]  # in minutes
#
#     for scenario in get_scenarios("scenarios_transient.csv"):
#         solver = rte(
#             scn2dict(scenario),
#             heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
#         )
#
#         # initial steady state
#         solver.args[VariableType.TRANSIT.value] = scenario["I_initial"]
#         solver.update()
#         initial_state = solver.steady_temperature()
#
#         # final steady state
#         solver.args[VariableType.TRANSIT.value] = scenario["I_final"]
#         solver.update()
#         final_state = solver.steady_temperature(
#             surface_temperature_guess=scenario["T_surf_transient_final"],
#             core_temperature_guess=scenario["T_heart_transient_final"],
#         )
#
#         # transient temperature (linearized)
#         transient_result = solver.transient_temperature_legacy(
#             offset=60 * np.array(offsets),
#             surface_temperature_0=initial_state[TemperatureLocation.SURFACE],
#             core_temperature_0=initial_state[TemperatureLocation.CORE],
#             time_constant=time_constant,
#         )
#
#         # check final temperature
#         assert np.isclose(
#             scenario["T_mean_final"],
#             final_state[TemperatureLocation.AVERAGE][0],
#         )
#
#         # check transient temperature
#         for offset in offsets:
#             for temperature in ["surface", "core", "mean"]:
#                 expected_temperature = scenario[f"T_{temperature}_{offset}"]
#                 # todo : recuperer mieux que ca : deja comprendre la forme de ce qui est renvoyé.
#                 result_temperature = transient_result[f"T_{temperature}"][
#                     np.argmin(np.abs(transient_result[VariableType.TIME] - offset * 60))
#                 ]
#                 assert np.isclose(expected_temperature, result_temperature)
