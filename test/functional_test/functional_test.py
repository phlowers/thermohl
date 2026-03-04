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
from thermohl.solver import rte, HeatEquationType
from thermohl.solver.enums.variable_type import VariableType


def get_cable_data(cable_name: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if cable_name in df["conductor"].values:
        return df[df["conductor"] == cable_name].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {cable_name} not found in file {f}.")


def get_scenarios():
    f = os.path.join("test", "functional_test", "scenarios.csv")
    return pd.read_csv(f, sep=";").to_dict(orient="records")


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

    print(f"YOMAN10  :{dic=}")
    return dic


# def test_steady_temperature():
#     for d in scenario("temperature", "steady"):
#         for _, e in d.items():
#             s = rte(
#                 scn2dict(e),
#                 heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
#             )
#             r = s.steady_temperature()
#
#             assert np.allclose(r[TemperatureLocation.SURFACE], e["T_surf"], atol=0.05)
#             assert np.allclose(r[TemperatureLocation.AVERAGE], e["T_mean"], atol=0.05)
#             assert np.allclose(r[TemperatureLocation.CORE], e["T_heart"], atol=0.05)


def test_steady_ampacity():
    for scenario in get_scenarios():
        solver = rte(
            scn2dict(scenario),
            heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
        )
        result = solver.steady_intensity(max_conductor_temperature=scenario["T_conf"])
        assert np.allclose(
            result[VariableType.TRANSIT], scenario["Ampacite"], atol=0.05
        )


# def test_transient_temperature():
#     atol = 0.5
#
#     # this is hard-coded, maybe it should be put in the yaml file ...
#     time_constant = 600.0
#     dt = 10.0
#     minute = 60
#
#     for d in scenario("temperature", "transient"):
#         for _, e in d.items():
#             # solver
#             s = rte(
#                 scn2dict(e),
#                 heat_equation=HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY,
#             )
#
#             # initial steady state
#             s.args[VariableType.TRANSIT.value] = e["I0_cable"]
#             s.update()
#             ri = s.steady_temperature()
#
#             # final steady state
#             s.args[VariableType.TRANSIT.value] = e["iac"]
#             s.update()
#             rf = s.steady_temperature(
#                 surface_temperature_guess=e["T_mean_final"],
#                 core_temperature_guess=e["T_mean_final"],
#             )
#
#             # time
#             time = np.arange(0.0, 1800.0, dt)
#
#             # transient temperature (linearized)
#             rl = s.transient_temperature_legacy(
#                 time=time,
#                 surface_temperature_0=ri[TemperatureLocation.SURFACE],
#                 core_temperature_0=ri[TemperatureLocation.CORE],
#                 time_constant=time_constant,
#             )
#
#             # check final temp
#             assert np.isclose(
#                 e["T_mean_final"], rf[TemperatureLocation.AVERAGE][0], atol=atol
#             )
#
#             # check transient temp
#             for k1, k2 in zip(
#                 ["T_surf_transient", "T_mean_transient", "T_heart_transient"],
#                 [
#                     TemperatureLocation.SURFACE,
#                     TemperatureLocation.AVERAGE,
#                     TemperatureLocation.CORE,
#                 ],
#             ):
#                 expected_time = np.array(list(e[k1].keys())) * minute
#                 expected_temp = np.array(list(e[k1].values()))
#                 estimated_temp = np.interp(expected_time, rl[VariableType.TIME], rl[k2])
#                 assert np.allclose(expected_temp, estimated_temp, atol=atol)
