# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from thermohl import solver
from thermohl.solver.enums.solver_type import SolverType
from thermohl.solver.enums.variable_type import VariableType

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    # Transit variation
    N = 1
    I0 = 300.0
    Im = 3000.0
    tau = 1000.0
    t = np.linspace(0.0, 7200.0, 721)
    transit = I0 * np.ones_like(tau) + (Im - I0) * (
        np.where(np.abs(1800 - t) <= tau, 1, 0)
        + np.where(np.abs(5400 - t) <= tau, 1, 0)
    )

    # Solver input and solver
    dct = dict(
        latitude=45.0,
        altitude=100.0,
        azimuth=90.0,
        month=3,
        day=21,
        hour=0,
        ambient_temperature=20.0,
        wind_speed=2.0,
        wind_angle=10,  # . * (1 + 0.5 * np.random.randn(len(t))),
        transit=np.nan,
    )

    # plot transit over time
    plt.figure()
    plt.plot(t, transit)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Transit (A)")

    plt.show()

    # dict with all 4 solvers
    kys = [
        SolverType.SOLVER_CIGRE,
        SolverType.SOLVER_IEEE,
        SolverType.SOLVER_RTE,
        SolverType.SOLVER_OLLA,
    ]
    slv = dict(
        cigre=solver.cigre(dct),
        ieee=solver.ieee(dct),
        rte=solver.rte(dct),
    )

    # solve and plot, add steady to check differences
    plt.figure()
    for i, key in enumerate(slv):
        elm = slv[key]
        elm.dc[VariableType.TRANSIT] = transit
        df = elm.steady_temperature()
        elm.dc[VariableType.TRANSIT] = np.nan
        cl = "C%d" % (i % 10,)
        T1 = df["T_surf"].values
        T2 = elm.transient_temperature(t, T0=np.array(T1[0]), transit=transit)["T_surf"]
        plt.plot(t, T1, "--", c=cl, label="%s - steady" % (key,))
        plt.plot(t, T2, "-", c=cl, label="%s - transient" % (key,))
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.legend()

    plt.show()

    # only rte but with core temp
    plt.figure()
    elm = slv["rte"]
    elm.dc[VariableType.TRANSIT] = transit
    df = elm.steady_temperature(return_avg=True, return_core=True)
    elm.dc[VariableType.TRANSIT] = np.nan
    cl = "C0"
    dg = elm.transient_temperature(
        t, T0=T1[0], transit=transit, return_avg=True, return_core=True
    )
    plt.fill_between(t, df["T_surf"], df["T_core"], fc=cl, alpha=0.33)
    plt.plot(t, df["T_avg"], "--", c=cl, label="%s - steady" % (key,))

    plt.fill_between(t, dg["T_surf"], dg["T_core"], fc=cl, alpha=0.33)
    plt.plot(t, dg["T_avg"], "-", c=cl, label="%s - transient" % (key,))
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.legend()

    plt.show()
