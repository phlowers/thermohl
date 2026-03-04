# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone
import numpy as np

from thermohl.power import ieee
from thermohl import solver


def test_compare_powers():
    """Compare computed values to hard-coded ones from ieee std 38-2012."""
    dic = solver.default_values()

    # there are a lot of rounding in the standard guide, hence the relatively
    # large tolerances used in our tests ...

    dic["wind_speed"] = 0.61
    dic["wind_azimuth"] = 0.0
    dic["emissivity"] = 0.8
    dic["solar_absorptivity"] = 0.8
    dic["ambient_temperature"] = 40.0
    dic["temp_high"] = 75.0
    dic["temp_low"] = 25.0
    dic["linear_resistance_temp_high"] = 8.688e-05
    dic["linear_resistance_temp_low"] = 7.283e-05
    dic["cable_azimuth"] = 90.0
    dic["latitude"] = 30.0
    dic["turbidity"] = 0.0
    dic["altitude"] = 0.0
    dic["outer_diameter"] = 28.14 * 1.0e-03
    dic["core_diameter"] = 10.4 * 1.0e-03
    dic["datetime_utc"] = datetime(2000, 6, 10, 11, tzinfo=timezone.utc)

    conductor_temperature = 100.0

    assert np.isclose(
        ieee.ConvectiveCooling(**dic).value(conductor_temperature), 81.93, rtol=0.002
    )
    assert np.isclose(
        ieee.RadiativeCooling(**dic).value(conductor_temperature), 39.1, rtol=0.001
    )
    assert np.isclose(
        ieee.SolarHeating(**dic).value(conductor_temperature), 22.44, rtol=0.006
    )
    joule_heating = ieee.JouleHeating(**dic)
    assert np.isclose(
        joule_heating._rdc(conductor_temperature), 9.390e-05, rtol=1.0e-09
    )

    # additional debug
    ieee.SolarHeating(**dic).value(conductor_temperature)

    from thermohl import sun

    sd = sun.solar_declination(dic["datetime_utc"].date())
    assert np.isclose(np.rad2deg(sd), 23.0, atol=0.1)

    ha = sun.hour_angle(sun.time_to_float_hours(dic["datetime_utc"].time()))
    assert np.isclose(np.rad2deg(ha), -15.0, atol=0.1)

    sa = sun.solar_altitude(
        np.deg2rad(dic["latitude"]),
        dic["datetime_utc"].date(),
        sun.time_to_float_hours(dic["datetime_utc"].time()),
    )
    assert np.isclose(np.rad2deg(sa), 74.8, atol=0.2)

    sz = sun.solar_azimuth(
        np.deg2rad(dic["latitude"]),
        dic["datetime_utc"].date(),
        sun.time_to_float_hours(dic["datetime_utc"].time()),
    )
    assert np.isclose(np.rad2deg(sz), 114.0, atol=0.5)

    th = np.arccos(np.cos(sa) * np.cos(sz - np.deg2rad(dic["cable_azimuth"])))
    assert np.isclose(np.rad2deg(th), 76.1, atol=1.5)
