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
    # f = "cable_catalog.csv"
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")


def scn2dict(d: dict) -> dict:
    dic = cable_data(d["cable"])
    for k in ["latitude", "longitude", "altitude"]:
        dic[k] = d[k]

    dic["Ta"] = d["weather_temperature"]
    dic["ws"] = d["wind_speed"]
    dic["wa"] = d["wind_angle"]
    dic["az"] = 90.0
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
    scn = yaml.safe_load(open("test/functional_test/scenario.yaml"))
    # scn = yaml.safe_load(open("scenario.yaml"))
    scn = scn["temperature"]["steady"]

    for d in scn:
        for _, e in d.items():
            s = cner(scn2dict(e), heateq="3tl")
            r = s.steady_temperature()

            assert np.allclose(r["t_surf"], e["T_surf"], atol=3.1)
            assert np.allclose(r["t_avg"], e["T_mean"], atol=3.1)
            assert np.allclose(r["t_core"], e["T_heart"], atol=3.1)


def test_steady_ampacity():
    scn = yaml.safe_load(open("test/functional_test/scenario.yaml"))
    # scn = yaml.safe_load(open("scenario.yaml"))
    scn = scn["ampacity"]["steady"]

    for d in scn:
        for _, e in d.items():
            s = cner(scn2dict(e), heateq="3tl")
            r = s.steady_intensity(T=e["Tmax_cable"])

            assert np.allclose(r["I"], e["I_max"], atol=40.)
