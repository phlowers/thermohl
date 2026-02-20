# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from thermohl import solver
from thermohl.power import olla


def plot_skin_effect():
    x = np.linspace(1.0, 50.0, 50)
    outer_diameter = 1.0
    R = 60 / x**2 / 1609.34  # Rdc from x and mile to meter conversion
    ra = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.76,
            0.80,
            0.82,
            0.84,
            0.86,
            0.88,
            0.90,
            0.92,
        ]
    )
    dl = ra * outer_diameter
    cl = cm.Spectral(np.linspace(0, 1, len(dl)))

    di = solver.default_values()

    plt.figure()
    for i, core_diameter in enumerate(dl):
        di.update(
            [
                ("outer_diameter", outer_diameter),
                ("core_diameter", core_diameter),
                ("linear_resistance_dc_20c", R),
                ("frequency", 50.0),
            ]
        )
        joule_heating = olla.JouleHeating(**di)
        plt.plot(
            x,
            joule_heating._ks(R),
            "-",
            c=cl[i],
            label="with $r_{in}/r_{out}$=%.2f" % (core_diameter,),
        )
    plt.grid(True)
    plt.legend()
    plt.ylim([1.00, 1.15])
    plt.title("_ks [from olla.Joule.Heating - computed with Bessel approx.]")

    plt.show()

    return


if __name__ == "__main__":
    """Test skin effect coefficient."""

    import matplotlib

    matplotlib.use("TkAgg")
    plt.close("all")

    plot_skin_effect()
