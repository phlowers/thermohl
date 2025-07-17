# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import matplotlib.pyplot as plt
import numpy as np

from thermohl import solver

def example_solver1t(dic):
    # create solver with ieee power terms and 1t heat equation; other options
    # available are solver.cigre, solver.olla and solver.rte for power terms
    slvr = solver.ieee(dic, heateq="1t")

    # test 1: compute temperature (using transit from dic["I"])
    dtemp = slvr.steady_temperature()

    # test 2 : compute max intensity
    Trep = 50.0
    damp = slvr.steady_intensity(Trep)

    # plot results
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(slvr.args.hour, dtemp['t'], c='C0', label='Conductor temperature (C)')
    ax[0].axhline(Trep, ls="--", c='C1', label="Maximum temperature for ampacity")
    ax[1].plot(slvr.args.hour, slvr.args.I, c='C0', label="Transit (A)")
    ax[1].plot(slvr.args.hour, damp['I'], c='C1', label='Conductor ampacity (A)')
    ax[2].plot(slvr.args.hour, dtemp['P_solar'], label="Solar heating power (W/m)")
    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()
    ax[2].set_xlabel("Hour")
    plt.show()


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    # Generate input dict (for the sake of simplicity, only a few inputs are
    # used, the rest is filled with default values).
    dic = dict(
        lat=46.1,
        alt=123.,
        azm=31.,
        month=6,
        day=20,
        hour=np.linspace(0., 23., 24),
        I=np.array([400. for i in range(12)] + [700. for i in range(12)]),
    )
    example_solver1t(dic)


