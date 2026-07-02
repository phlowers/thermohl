# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from math import pi
import pytest

import numpy as np
from thermohl.power.rte.solar_heating import (
    compute_solar_irradiance,
    SolarHeating,
    diffuse_and_beam_radiations,
    estimate_nebulosity,
    solar_irradiance,
    estimate_nebulosity_from_diffuse_and_beam_radiation,
    compute_global_radiation,
    compute_diffuse_radiation,
    compute_beam_radiation,
)


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
    datetime_utc = np.array(
        [
            np.datetime64("2026-03-09T08:50:00"),
            np.datetime64("2026-03-09T08:50:00"),
            np.datetime64("2026-03-09T08:50:00"),
        ]
    )

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
    datetime_utc = np.array(
        [
            np.datetime64("2026-03-09T08:50:00"),
            np.datetime64("2026-03-09T08:50:00"),
            np.datetime64("2026-03-09T08:50:00"),
        ]
    )

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


def test_diffuse_and_beam_radiations() -> None:
    datetime_utc = np.array(
        [
            np.datetime64("2026-06-15T00:00"),
            np.datetime64("2026-06-15T06:00"),
            np.datetime64("2026-06-15T12:00"),
        ]
    )
    latitude = np.array([48, 48, 48])
    longitude = np.array([21, 21, 21])
    nebulosity = np.array([0, 2, 8])
    diffuse_radiation, beam_radiation = diffuse_and_beam_radiations(
        datetime_utc,
        latitude,
        longitude,
        nebulosity,
    )
    assert np.allclose(diffuse_radiation, np.array([0, 149.44519, 189.98370]))
    assert np.allclose(beam_radiation, np.array([0, 555.11970, 0]))


def test_diffuse_and_beam_radiations__wrong_nebulosity() -> None:
    datetime_utc = np.array(
        [
            np.datetime64("2026-06-15T00:00"),
            np.datetime64("2026-06-15T06:00"),
            np.datetime64("2026-06-15T12:00"),
        ]
    )
    latitude = np.array([48, 48, 48])
    longitude = np.array([21, 21, 21])
    nebulosity = np.array([0, 2, 8.5])
    with pytest.raises(ValueError):
        diffuse_and_beam_radiations(
            datetime_utc,
            latitude,
            longitude,
            nebulosity,
        )


@pytest.mark.parametrize(
    "input_nebulosity, solar_altitude, expected_nebulosity",
    [
        (0, -pi / 4, np.nan),
        (0, pi / 3, 0),
        (0, pi / 2, 0),
        (0, pi, np.nan),
        (0, 5 * pi / 4, np.nan),
        (3.5, -pi / 4, np.nan),
        (3.5, pi / 3, 3),
        (3.5, pi / 2, 3),
        (3.5, pi, np.nan),
        (3.5, 5 * pi / 4, np.nan),
        (8, -pi / 4, np.nan),
        (8, pi / 3, 8),
        (8, pi / 2, 8),
        (8, pi, np.nan),
        (8, 5 * pi / 4, np.nan),
    ],
)
def test_estimate_nebulosity_from_diffuse_and_beam_radiation__scalar(
    input_nebulosity,
    solar_altitude,
    expected_nebulosity,
) -> None:
    global_radiation = compute_global_radiation(solar_altitude, input_nebulosity)
    diffuse_radiation = compute_diffuse_radiation(global_radiation, input_nebulosity)
    beam_radiation = compute_beam_radiation(
        global_radiation, diffuse_radiation, solar_altitude
    )

    nebulosity_estimate = estimate_nebulosity_from_diffuse_and_beam_radiation(
        solar_altitude, diffuse_radiation + beam_radiation
    )

    np.testing.assert_allclose(nebulosity_estimate, expected_nebulosity)


def test_estimate_nebulosity_from_diffuse_and_beam_radiation__array() -> None:
    input_nebulosity = np.array([1, 2])
    solar_altitude = np.array([np.pi / 2, 0])
    expected_nebulosity = np.array([1, np.nan])

    global_radiation = compute_global_radiation(solar_altitude, input_nebulosity)
    diffuse_radiation = compute_diffuse_radiation(global_radiation, input_nebulosity)
    beam_radiation = compute_beam_radiation(
        global_radiation, diffuse_radiation, solar_altitude
    )

    nebulosity_estimate = estimate_nebulosity_from_diffuse_and_beam_radiation(
        solar_altitude, diffuse_radiation + beam_radiation
    )

    np.testing.assert_allclose(nebulosity_estimate, expected_nebulosity)


def test_estimate_nebulosity_from_diffuse_and_beam_radiation__no_solution(
    caplog,
) -> None:
    # Take the solar altitude and nebulosity which give the highest radiation (zenith and no clouds).
    # Compute the global radiation, and use a greater value to try to compute nebulosity.
    # It's impossible to find a nebulosity with the given radiation, attempting to
    # calculate it should raise a warning log and return a saturated value (0 in this case).
    solar_altitude = np.pi / 2

    global_radiation = (
        compute_global_radiation(solar_altitude=solar_altitude, nebulosity=0) + 10
    )
    diffuse_radiation = compute_diffuse_radiation(global_radiation, nebulosity=0)
    beam_radiation = compute_beam_radiation(
        global_radiation, diffuse_radiation, solar_altitude
    )

    with caplog.at_level("WARNING"):
        nebulosity = estimate_nebulosity_from_diffuse_and_beam_radiation(
            solar_altitude, diffuse_radiation + beam_radiation
        )
    assert "Bisection method" in caplog.text
    assert "do not satisfy convergence conditions" in caplog.text
    assert nebulosity == 0


def test_estimate_nebulosity__array() -> None:
    diffuse_plus_beam_radiation = np.array([700, 600])
    datetime_utc = np.array(
        [
            np.datetime64("2026-06-15T12:00:00"),
            np.datetime64("2026-06-15T12:00:00"),
        ]
    )
    latitude = np.array([45.0, 45.0])
    longitude = np.array([20.0, 20.0])
    nebulosity = estimate_nebulosity(
        diffuse_plus_beam_radiation,
        datetime_utc,
        latitude,
        longitude,
    )
    assert np.allclose(nebulosity, [5, 6])


def test_estimate_nebulosity__inconsistent_radiation() -> None:
    diffuse_plus_beam_radiation = np.array([10, 4200])
    datetime_utc = np.array([np.datetime64("2026-06-15T12:00:00")])
    latitude = np.array([45.0])
    longitude = np.array([20.0])
    nebulosity = estimate_nebulosity(
        diffuse_plus_beam_radiation,
        datetime_utc,
        latitude,
        longitude,
    )
    assert np.allclose(nebulosity, [8, 0])


def test_estimate_nebulosity__array_night() -> None:
    diffuse_plus_beam_radiation = np.array([700])
    datetime_utc = np.array([np.datetime64("2026-06-15T00:00:00")])
    latitude = np.array([45.0])
    longitude = np.array([20.0])
    nebulosity = estimate_nebulosity(
        diffuse_plus_beam_radiation,
        datetime_utc,
        latitude,
        longitude,
    )
    assert np.isnan(nebulosity[0])


def test_solar_irradiance() -> None:
    result = solar_irradiance(
        datetime_utc=np.datetime64("2026-07-01T14:30:00"),
        latitude=np.array([48]),
        longitude=np.array([-5]),
        nebulosity=np.array([5]),
        cable_azimuth=np.array([180]),
        albedo=np.array([0.12]),
    )
    assert np.allclose(result, np.array([957.73]))


@pytest.mark.parametrize(
    "datetime_utc, latitude, longitude, nebulosity, cable_azimuth, expected_result",
    [
        (
            np.array(
                [
                    np.datetime64("2026-07-01T14:30:00"),
                    np.datetime64("2026-07-01T14:30:00"),
                ]
            ),
            np.array([48, 49]),
            np.array([-5, 8]),
            np.array([0, 8]),
            np.array([0, 360]),
            np.array([1082.90, 284.73]),
        ),
        (
            np.array([np.datetime64("2026-07-01T01:00:30")]),
            np.array([49]),
            np.array([8]),
            np.array([8]),
            np.array([360]),
            np.array([0]),
        ),
    ],
    ids=[
        "Nominal",
        "Night",
    ],
)
def test_solar_irradiance__default_albedo(
    datetime_utc, latitude, longitude, nebulosity, cable_azimuth, expected_result
) -> None:
    result = solar_irradiance(
        datetime_utc,
        latitude,
        longitude,
        nebulosity,
        cable_azimuth,
    )
    assert np.allclose(result, expected_result)
