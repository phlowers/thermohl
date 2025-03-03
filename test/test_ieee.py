# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power import ieee
from thermohl import solver


def test_compare_powers():
    """Compare computed values to hard-coded ones from ieee std 38-2012."""
    dic = solver.default_values()

    # there are a lot of rounding in the standard guide, hence the relatively
    # large tolerances used in our tests ...

    dic["ws"] = 0.61
    dic["wa"] = 0.0
    dic["epsilon"] = 0.8
    dic["alpha"] = 0.8
    dic["Ta"] = 40.0
    dic["THigh"] = 75.0
    dic["TLow"] = 25.0
    dic["RDCHigh"] = 8.688e-05
    dic["RDCLow"] = 7.283e-05
    dic["azm"] = 90.0
    dic["lat"] = 30.0
    dic["tb"] = 0.0
    dic["alt"] = 0.0
    dic["D"] = 28.14 * 1.0e-03
    dic["d"] = 10.4 * 1.0e-03
    dic["month"] = 6
    dic["day"] = 10
    dic["hour"] = 11.0

    T = 100.0

    assert np.isclose(ieee.ConvectiveCooling(**dic).value(T), 81.93, rtol=0.002)
    assert np.isclose(ieee.RadiativeCooling(**dic).value(T), 39.1, rtol=0.001)
    assert np.isclose(ieee.SolarHeating(**dic).value(T), 22.44, rtol=0.001)
    jh = ieee.JouleHeating(**dic)
    assert np.isclose(jh._rdc(T), 9.390e-05, rtol=1.0e-09)

    # additional debug
    ieee.SolarHeating(**dic).value(T)

    from thermohl import sun

    sd = sun.solar_declination(dic["month"], dic["day"])
    assert np.isclose(np.rad2deg(sd), 23.0, rtol=0.001)

    ha = sun.hour_angle(dic["hour"], minute=0.0, second=0.0)
    assert np.isclose(np.rad2deg(ha), -15.0)

    sa = sun.solar_altitude(
        np.deg2rad(dic["lat"]), dic["month"], dic["day"], dic["hour"]
    )
    assert np.isclose(np.rad2deg(sa), 74.8, rtol=0.002)

    sz = sun.solar_azimuth(
        np.deg2rad(dic["lat"]), dic["month"], dic["day"], dic["hour"]
    )
    np.isclose(np.rad2deg(sz), 114.0, rtol=0.001)

    th = np.arccos(np.cos(sa) * np.cos(sz - dic["azm"]))
    np.isclose(np.rad2deg(th), 76.1, rtol=0.02)
