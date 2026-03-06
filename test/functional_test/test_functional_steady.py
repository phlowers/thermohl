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
from thermohl.solver import rte
from thermohl.solver.entities import VariableType, HeatEquationType, TemperatureType


def get_cable_data(cable_name: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if cable_name in df["conductor"].values:
        return df[df["conductor"] == cable_name].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {cable_name} not found in file {f}.")


def get_scenarios(scenario_file_name: str):
    f = os.path.join("test", "functional_test", scenario_file_name)
    df = pd.read_csv(f, sep=";")
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    return records


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
    if "Intensite" in scn:
        dic[VariableType.TRANSIT.value] = scn["Intensite"]

    return dic


def test_steady_temperature():
    for scenario in get_scenarios("scenarios_steady.csv"):
        solver = rte(
            scn2dict(scenario),
            heat_equation=HeatEquationType.THREE_TEMPERATURES_LEGACY,
        )
        result = solver.steady_temperature()
        assert np.allclose(
            result[TemperatureType.SURFACE], scenario["T_surf"], atol=0.05
        )
        assert np.allclose(
            result[TemperatureType.AVERAGE], scenario["T_moy"], atol=0.05
        )
        assert np.allclose(result[TemperatureType.CORE], scenario["T_coeur"], atol=0.05)


def test_steady_ampacity():
    for scenario in get_scenarios("scenarios_steady.csv"):
        solver = rte(
            scn2dict(scenario),
            heat_equation=HeatEquationType.THREE_TEMPERATURES_LEGACY,
        )
        result = solver.steady_intensity(max_conductor_temperature=scenario["T_conf"])
        assert np.allclose(
            result[VariableType.TRANSIT], scenario["Ampacite"], atol=0.05
        )
