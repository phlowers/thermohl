# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from thermohl.power.rte.solar_heating import compute_solar_irradiance, SolarHeating
from pandas import Timestamp


def test_compute_solar_irradiance_night():
    # When solar_altitude <= 0, irradiance should be 0
    res = compute_solar_irradiance(
        global_radiation=500.0,
        solar_altitude=-0.1,
        incidence=0.5,
        nebulosity=4.0,
        albedo=0.2,
    )
    assert res == 0.0

    res = compute_solar_irradiance(
        global_radiation=500.0,
        solar_altitude=0.0,
        incidence=0.5,
        nebulosity=4.0,
        albedo=0.2,
    )
    assert res == 0.0


def test_compute_solar_irradiance_day():
    global_radiation = 800.0
    solar_altitude = 0.5  # radians
    incidence = 0.8  # radians
    nebulosity = 2.0
    albedo = 0.15

    # Manual calculation
    # diffuse_radiation = 800 * (0.3 + 0.7 * (2/8)**2) = 275.0
    # beam_radiation = (800 - 275) / sin(0.5) = 251.6984077672066
    # solar_irradiance = beam_radiation * (sin(0.8) + pi/2 * 0.15 * sin(0.5)) + 275 * pi/2 * (1 + 0.15) = 1406.012913525969
    expected = 1406.012913525969

    res = compute_solar_irradiance(
        global_radiation=global_radiation,
        solar_altitude=solar_altitude,
        incidence=incidence,
        nebulosity=nebulosity,
        albedo=albedo,
    )

    assert np.isclose(res, expected)


def test_compute_solar_irradiance_array():
    global_radiation = np.array([700.0, 400.0, 600.0])
    solar_altitude = np.array([0.4, -0.1, 0.0])
    incidence = np.array([0.6, 0.2, 0.3])
    nebulosity = np.array([3.0, 5.0, 4.0])
    albedo = np.array([0.1, 0.2, 0.15])

    res = compute_solar_irradiance(
        global_radiation=global_radiation,
        solar_altitude=solar_altitude,
        incidence=incidence,
        nebulosity=nebulosity,
        albedo=albedo,
    )

    assert len(res) == 3
    assert res[1] == 0.0
    assert res[2] == 0.0

    # Check first element
    # diffuse = 700 * (0.3 + 0.7 * (3 / 8) ** 2) = 278.90625
    # beam = (700 - 278.90625) / sin(0.4) = 1081.3340623351336
    # expected0 = beam * (sin(0.6) + pi / 2 * 0.1 * sin(0.4)) + diffuse * pi / 2 * (1.1) = 1158.648723646428
    expected0 = 1158.631321677975
    assert np.isclose(res[0], expected0)


def test_solar_heating():
    latitude = [0.86892843, 0.86909212, 0.86957649]
    longitude = [0.03194659, 0.03498268, 0.03403367]
    cable_azimuth = [1.29034491, -2.43771926, 1.05803243]
    datetime_utc = [
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
    ]

    solar_heating = SolarHeating(
        latitude=latitude,
        longitude=longitude,
        cable_azimuth=cable_azimuth,
        datetime_utc=datetime_utc,
        outer_diameter=0,
        solar_absorptivity=0,
        albedo=0.15,
        nebulosity=0,
        measured_global_radiation=np.nan,
    )

    print(solar_heating.solar_irradiance)


def test_solar_irradiance_ignored_by_rte_solar_heating():
    latitude = [0.86892843, 0.86909212, 0.86957649]
    longitude = [0.03194659, 0.03498268, 0.03403367]
    cable_azimuth = [1.29034491, -2.43771926, 1.05803243]
    datetime_utc = [
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
        Timestamp("2026-03-09 08:50:00+0000", tz="UTC"),
    ]

    solar_heating_1 = SolarHeating(
        solar_irradiance=[0.0, 0.0, 0.0],  # must be ignored
        latitude=latitude,
        longitude=longitude,
        cable_azimuth=cable_azimuth,
        datetime_utc=datetime_utc,
        outer_diameter=0.03,
        solar_absorptivity=0.5,
        albedo=0.15,
        nebulosity=0,
        measured_global_radiation=np.nan,
    )

    solar_heating_2 = SolarHeating(
        solar_irradiance=[100, 200, 300],  # must also be ignored
        latitude=latitude,
        longitude=longitude,
        cable_azimuth=cable_azimuth,
        datetime_utc=datetime_utc,
        outer_diameter=0.03,
        solar_absorptivity=0.5,
        albedo=0.15,
        nebulosity=0,
        measured_global_radiation=np.nan,
    )

    # The provided solar_irradiance keyword argument must be ignored.
    assert np.allclose(
        solar_heating_1.solar_irradiance, solar_heating_2.solar_irradiance
    )

    # With non-zero parameters, value() would differ if the keyword argument were used.
    assert np.allclose(solar_heating_1.value(100), solar_heating_2.value(100))
