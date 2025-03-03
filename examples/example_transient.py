# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermohl import solver

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
    I = I0 * np.ones_like(tau) + (Im - I0) * (
        np.where(np.abs(1800 - t) <= tau, 1, 0)
        + np.where(np.abs(5400 - t) <= tau, 1, 0)
    )

    # Solver input and solver
    dct = dict(
        lat=45.0,
        alt=100.0,
        azm=90.0,
        month=3,
        day=21,
        hour=0,
        Ta=20.0,
        ws=2.0,
        wa=10,  # . * (1 + 0.5 * np.random.randn(len(t))),
        I=np.nan,
    )

    # plot transit over time
    plt.figure()
    plt.plot(t, I)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Transit (A)")

    plt.show()

    # dict with all 4 solvers
    kys = ["cigre", "ieee", "rte", "rtem"]
    slv = dict(
        cigre=solver.cigre(dct),
        ieee=solver.ieee(dct),
        cner=solver.cner(dct),
    )

    # solve and plot, add steady to check differences
    plt.figure()
    for i, key in enumerate(slv):
        elm = slv[key]
        elm.dc["I"] = I
        df = elm.steady_temperature()
        elm.dc["I"] = np.nan
        cl = "C%d" % (i % 10,)
        T1 = df["T_surf"].values
        T2 = elm.transient_temperature(t, T0=np.array(T1[0]), transit=I)["T_surf"]
        plt.plot(t, T1, "--", c=cl, label="%s - steady" % (key,))
        plt.plot(t, T2, "-", c=cl, label="%s - transient" % (key,))
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.legend()

    plt.show()

    # only rte but with core temp
    plt.figure()
    elm = slv["cner"]
    elm.dc["I"] = I
    df = elm.steady_temperature(return_avg=True, return_core=True)
    elm.dc["I"] = np.nan
    cl = "C0"
    dg = elm.transient_temperature(
        t, T0=T1[0], transit=I, return_avg=True, return_core=True
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
