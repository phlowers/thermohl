# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import datetime
import os.path
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

from thermohl.solver import rte


def cable_data(s: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if s in df["conductor"].values:
        return df[df["conductor"] == s].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {s} not found in file {f}.")


def scenario(key: str, mode: str) -> List[Dict]:
    f = os.path.join("test", "functional_test", "scenario.yaml")
    y = yaml.safe_load(open(f))
    return y[key][mode]


def scn2dict(d: dict) -> dict:
    """Convert scenario to thermohl input."""

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
    for d in scenario("temperature", "steady"):
        for _, e in d.items():
            s = rte(scn2dict(e), heateq="3tl")
            r = s.steady_temperature(return_power=False)

            assert np.allclose(r["t_surf"], e["T_surf"], atol=0.05)
            assert np.allclose(r["t_avg"], e["T_mean"], atol=0.05)
            assert np.allclose(r["t_core"], e["T_heart"], atol=0.05)


def test_steady_temperature_batch():
    din = []
    dref = []
    for d in scenario("temperature", "steady"):
        for _, e in d.items():
            din.append(pd.DataFrame(scn2dict(e), index=(0,)))
            dref.append(
                pd.DataFrame(
                    {k: e[k] for k in ("T_surf", "T_mean", "T_heart") if k in e},
                    index=(0,),
                )
            )
    din = pd.concat(din).reset_index(drop=True)
    dref = pd.concat(dref).reset_index(drop=True)
    dref.rename(
        columns={"T_surf": "t_surf", "T_mean": "t_avg", "T_heart": "t_core"},
        inplace=True,
    )

    s = rte(din, heateq="3tl")
    r = s.steady_temperature(return_power=False)
    assert np.allclose(r.values, dref.values, atol=0.05)


def test_steady_ampacity():
    for d in scenario("ampacity", "steady"):
        for _, e in d.items():
            s = rte(scn2dict(e), heateq="3tl")
            r = s.steady_intensity(T=e["Tmax_cable"])

            assert np.allclose(r["I"], e["I_max"], atol=0.05)


def test_steady_ampacity_batch():
    din = []
    dref = []
    for d in scenario("ampacity", "steady"):
        for _, e in d.items():
            din.append(pd.DataFrame(scn2dict(e), index=(0,)))
            dref.append(
                pd.DataFrame(
                    {k: e[k] for k in ("Tmax_cable", "I_max") if k in e},
                    index=(0,),
                )
            )
    din = pd.concat(din).reset_index(drop=True)
    dref = pd.concat(dref).reset_index(drop=True)

    s = rte(din, heateq="3tl")
    r = s.steady_intensity(T=dref["Tmax_cable"], return_power=False)
    assert np.allclose(r["I"].values, dref["I_max"].values, atol=0.05)


def test_transient_temperature():

    atol = 0.1

    # this is hard-coded, maybe it should be put in the yaml file ...
    tau = 600.0
    dt = 10.0
    minute = 60

    for d in scenario("temperature", "transient"):
        for _, e in d.items():

            # solver
            s = rte(scn2dict(e), heateq="3tl")

            # initial steady state
            s.args["I"] = e["I0_cable"]
            s.update()
            ri = s.steady_temperature()

            # final steady state
            s.args["I"] = e["iac"]
            s.update()
            rf = s.steady_temperature(Tsg=e["T_mean_final"], Tcg=e["T_mean_final"])

            # time
            time = np.arange(0.0, 1800.0 + dt, dt)

            # transient temperature (linearized)
            rl = s.transient_temperature_legacy(
                time=time, Ts0=ri["t_surf"], Tc0=ri["t_core"], tau=tau
            )

            # check final temp
            assert np.isclose(e["T_mean_final"], rf["t_avg"][0], atol=atol)

            # check transient temp
            for k1, k2 in zip(
                ["T_surf_transient", "T_mean_transient", "T_heart_transient"],
                ["t_surf", "t_avg", "t_core"],
            ):
                expected_time = np.array(list(e[k1].keys())) * minute
                expected_temp = np.array(list(e[k1].values()))
                estimated_temp = np.interp(expected_time, rl["time"], rl[k2])
                assert np.allclose(expected_temp, estimated_temp, atol=atol)


def test_transient_temperature_batch():

    atol = 0.1

    # this is hard-coded, maybe it should be put in the yaml file ...
    tau = 600.0
    dt = 10.0
    minute = 60

    din = []
    I0 = []
    dref = {"time": [], "t_surf": [], "t_avg": [], "t_core": [], "t_avg_final": []}
    for d in scenario("temperature", "transient"):
        for _, e in d.items():
            din.append(pd.DataFrame(scn2dict(e), index=(0,)))
            I0.append(e["I0_cable"])
            for k1, k2 in zip(
                ["T_surf_transient", "T_mean_transient", "T_heart_transient"],
                ["t_surf", "t_avg", "t_core"],
            ):
                dref[k2].append(np.array(list(e[k1].values())))
            dref["t_avg_final"].append(e["T_mean_final"])
    din = pd.concat(din).reset_index(drop=True)
    I0 = np.array(I0)
    dref["time"] = np.array(list(e["T_surf_transient"].keys())) * minute * 1.0
    for k in ["t_surf", "t_avg", "t_core"]:
        dref[k] = np.array(dref[k]).T
    dref["t_avg_final"] = np.array(dref["t_avg_final"])

    # solver
    s = rte(din, heateq="3tl")

    # initial steady state
    s.args["I"] = I0
    s.update()
    ri = s.steady_temperature(return_power=False)

    # final steady state
    s.args["I"] = din["I"]
    s.update()
    rf = s.steady_temperature(Tsg=dref["t_avg_final"], Tcg=dref["t_avg_final"])

    # check final temp
    assert np.allclose(dref["t_avg_final"], rf["t_avg"].values, atol=atol)

    # time
    time = np.arange(0.0, 1800.0 + dt, dt)

    # transient temperature (linearized)
    rl = s.transient_temperature_legacy(
        time=time, Ts0=ri["t_surf"], Tc0=ri["t_core"], tau=tau
    )

    # filter expected times
    ix = np.in1d(rl["time"], dref["time"]).nonzero()[0]
    for k in ["t_surf", "t_avg", "t_core"]:
        assert np.allclose(dref[k], rl[k][ix, :], atol=atol)
