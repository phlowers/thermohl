# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import datetime
import os.path

import numpy as np
import pandas as pd
import yaml

from thermohl.solver import cner


def cable_data(s: str) -> dict:
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")


def scn2dict(d: dict) -> dict:
    dic = cable_data(d["cable"])

    dic["lat"] = d["latitude"]
    dic["lon"] = d["longitude"]
    dic["alt"] = d["altitude"]
    dic["azm"] = 90.0
    dic["Ta"] = d["weather_temperature"]
    dic["ws"] = d["wind_speed"]
    # in scenario file, wind angles are given regarding span, where in thermohl
    # they are supposed to be regarding north, hence this conversion formula
    dic["wa"] = np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(np.abs(dic["azm"] - d["wind_angle"]) % 180.0)))
    )
    dic["alpha"] = 0.9
    dic["epsilon"] = 0.8

    dt = datetime.datetime.fromisoformat(d["date"])
    dic["month"] = dt.month
    dic["day"] = dt.day
    dic["hour"] = (
        dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond * 1.0e-06) / 3600.0
    )
    if "iac" in d.keys():
        dic["I"] = d["iac"]

    return dic


def test_steady_temperature():
    f = os.path.join("test", "functional_test", "scenario.yaml")
    scn = yaml.safe_load(open(f))
    scn = scn["temperature"]["steady"]

    for d in scn:
        for _, e in d.items():
            s = cner(scn2dict(e), heateq="3tl")
            r = s.steady_temperature()

            assert np.allclose(r["t_surf"], e["T_surf"], atol=0.05)
            assert np.allclose(r["t_avg"], e["T_mean"], atol=0.05)
            assert np.allclose(r["t_core"], e["T_heart"], atol=0.05)


def test_steady_ampacity():
    f = os.path.join("test", "functional_test", "scenario.yaml")
    scn = yaml.safe_load(open(f))
    scn = scn["ampacity"]["steady"]

    for d in scn:
        for _, e in d.items():
            s = cner(scn2dict(e), heateq="3tl")
            r = s.steady_intensity(T=e["Tmax_cable"])

            assert np.allclose(r["I"], e["I_max"], atol=0.05)
