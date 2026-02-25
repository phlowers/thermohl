# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import calendar
import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermohl import solver
from thermohl.power import cigre
from thermohl.power import rte
from thermohl.power import ieee
from thermohl.power import olla


def plot_joule_heating(dic):
    mdl = [
        dict(label="cigre", model=cigre.JouleHeating(**dic)),
        dict(label="rte", model=rte.JouleHeating(**dic)),
        dict(label="ieee", model=ieee.JouleHeating(**dic)),
        dict(label="olla", model=olla.JouleHeating(**dic)),
    ]
    plt.figure()
    t = np.linspace(0.0, 200.0, 4001)
    for d in mdl:
        plt.plot(t, d["model"].value(t), label=d["label"])
    plt.grid(True)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Joule Heating per unit length (Wm**-1)")
    plt.legend()
    plt.show()

    return


def plot_solar_heating(dic):
    mdl = [
        dict(label="cigre", cm=cm.spring, model=cigre.SolarHeating(**dic)),
        dict(label="rte", cm=cm.summer, model=rte.SolarHeating(**dic)),
        dict(label="ieee", cm=cm.autumn, model=ieee.SolarHeating(**dic)),
        dict(label="olla", cm=cm.winter, model=olla.SolarHeating(**dic)),
    ]

    # examples 1
    month = np.array(range(1, 13))
    day = 21 * np.ones_like(month)
    hour = np.linspace(0, 24, 24 * 120 + 1)
    fig, ax = plt.subplots(nrows=1, ncols=len(mdl))
    for k in range(12):
        dic.update([("month", month[k]), ("day", day[k]), ("hour", hour)])
        for i, d in enumerate(mdl):
            d["model"].__init__(**dic)
            c = d["cm"](np.linspace(0.0, 1.0, len(day) + 2))[1:-1]
            ax[i].plot(
                hour,
                d["model"].value(0.0),
                "-",
                c=c[k],
                label=calendar.month_name[k + 1],
            )
            ax[i].set_title(d["label"])

    ax[0].set_ylabel("Solar Heating per unit length (Wm**-1)")
    for j in range(len(mdl)):
        ax[j].set_ylim([0, 25])
        ax[j].grid(True)
        ax[j].legend()
        ax[j].set_xlabel("Hour of day")

    # examples 2
    dr = pd.date_range(
        datetime.datetime(2001, 1, 1), datetime.datetime(2001, 12, 31), freq="h"
    )
    hl = np.linspace(0, 24, 13)[:-1]
    fig, ax = plt.subplots(nrows=1, ncols=len(mdl))
    for k, h in enumerate(hl):
        dic.update([("month", dr.month.values), ("day", dr.day.values), ("hour", h)])
        for i, d in enumerate(mdl):
            d["model"].__init__(**dic)
            c = d["cm"](np.linspace(0.0, 1.0, len(hl) + 2))[1:-1]
            ax[i].plot(
                dr, d["model"].value(0.0), "-", c=c[k], label="At %02d:00" % (h,)
            )
            ax[i].set_title(d["label"])
    ax[0].set_ylabel("Solar Heating per unit length (Wm**-1)")
    for j in range(len(mdl)):
        ax[j].set_ylim([0, 25])
        ax[j].grid(True)
        ax[j].legend()
        ax[j].set_xlabel("Day of year")

    plt.show()

    return


def plot_convective_cooling(dic):
    wind_speed = np.linspace(0, 1, 5)
    wind_angle = dic["cable_azimuth"] - np.array([0, 45, 90])
    conductor_temperature = np.linspace(0.0, 80.0, 41)
    ambient_temperatures = np.linspace(-10, 40, 6)

    # olla not tested here since olla's convective cooling is the same as ieee's one
    mdl = [
        dict(label="cigre", cm=cm.spring, model=cigre.ConvectiveCooling(**dic)),
        dict(label="rte", cm=cm.summer, model=rte.ConvectiveCooling(**dic)),
        dict(label="ieee", cm=cm.autumn, model=ieee.ConvectiveCooling(**dic)),
        dict(label="olla", cm=cm.winter, model=olla.ConvectiveCooling(**dic)),
    ]

    fig, ax = plt.subplots(nrows=len(wind_speed), ncols=len(wind_angle))
    for i, u in enumerate(wind_speed):
        for j, a in enumerate(wind_angle):
            for m, ambient_temperature in enumerate(ambient_temperatures):
                dic.update(
                    [
                        ("wind_speed", u),
                        ("wind_angle", a),
                        ("ambient_temperature", ambient_temperature),
                    ]
                )
                for k, d in enumerate(mdl):
                    d["model"].__init__(**dic)
                    c = d["cm"](np.linspace(0.0, 1.0, len(ambient_temperatures) + 2))[
                        1:-1
                    ]
                    ax[i, j].plot(
                        conductor_temperature,
                        d["model"].value(conductor_temperature),
                        "-",
                        c=c[m],
                        label="T$_a$=%.0f C" % (ambient_temperature,),
                    )
                ax[i, j].set_title("At u=%.1f and $\phi$=%.0f" % (u, a))
                ax[i, j].grid(True)
                ax[i, j].set_ylim([-50, 100])
    for i in range(len(wind_speed)):
        ax[i, 0].set_ylabel("Convective Cooling per unit length (Wm**-1)")
    for j in range(len(wind_angle)):
        ax[-1, j].set_xlabel("Conductor temperature (C)")
    ax[0, -1].legend()

    plt.show()

    return


def plot_radiative_cooling(dic):
    conductor_temperature = np.linspace(0.0, 80.0, 41)
    ambient_temperatures = np.linspace(-20, 50, 8)
    cl = cm.Spectral_r(np.linspace(0.0, 1.0, len(ambient_temperatures) + 2)[1:-1])

    # rte is not displayed since it is the same as ieee

    plt.figure()
    plt.plot(np.nan, np.nan, ls="-", c="gray", label="cigre")
    plt.plot(np.nan, np.nan, ls="--", c="gray", label="ieee")
    plt.plot(np.nan, np.nan, ls=":", c="gray", label="olla")

    for i, ambient_temperature in enumerate(ambient_temperatures):
        dic["ambient_temperature"] = ambient_temperature

        radiative_cooling = cigre.RadiativeCooling(**dic)
        plt.plot(
            conductor_temperature,
            radiative_cooling.value(conductor_temperature),
            ls="-",
            c=cl[i],
            label="T$_a$=%.0f C" % (ambient_temperature,),
        )
        radiative_cooling = ieee.RadiativeCooling(**dic)
        plt.plot(
            conductor_temperature,
            radiative_cooling.value(conductor_temperature),
            ls="--",
            c=cl[i],
        )

        radiative_cooling = olla.RadiativeCooling(**dic)
        plt.plot(
            conductor_temperature,
            radiative_cooling.value(conductor_temperature),
            ":",
            c=cl[i],
        )

    plt.grid(True)
    plt.xlabel("Conductor temperature (C)")
    plt.ylabel("Radiative Cooling per unit length (Wm**-1)")
    plt.legend()

    plt.show()

    return


if __name__ == "__main__":
    """Check all power terms using default values. We use 45 deg for latitude
    as the 0 in default does not show much variation for solar_heating."""

    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    dic = solver.default_values()
    dic["latitude"] = 45.0
    dic["turbidity"] = 0.33
    dic["albedo"] = 0.85

    plot_joule_heating(dic)
    plot_solar_heating(dic)
    plot_convective_cooling(dic)
    plot_radiative_cooling(dic)
