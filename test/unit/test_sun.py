# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from thermohl.sun import utc2solar_hour


def test_scalar_input():
    # Test with scalar inputs
    hour = 12
    minute = 30
    second = 45
    lon = np.deg2rad(45)  # 45 degrees east

    result = utc2solar_hour(hour, minute, second, lon)
    expected_result = 12 + 30 / 60.0 + 45 / 3600.0 + 45 / 15.0
    assert np.isclose(result, expected_result)


def test_array_input():
    # Test with numpy array inputs
    hours = np.array([12, 15, 18])
    minutes = np.array([30, 45, 0])
    seconds = np.array([45, 30, 15])
    lons = np.deg2rad(np.array([45, 90, 135]))  # 45, 90, and 135 degrees east

    result = utc2solar_hour(hours, minutes, seconds, lons)
    expected_result = np.array(
        [
            12 + 30 / 60.0 + 45 / 3600.0 + 45 / 15.0,
            15 + 45 / 60.0 + 30 / 3600.0 + 90 / 15.0,
            18 + 0 / 60.0 + 15 / 3600.0 + 135 / 15.0,
        ]
    )
    np.testing.assert_array_almost_equal(result, expected_result)


def test_edge_cases():
    # Test edge cases
    hour = 23
    minute = 59
    second = 59
    lon = np.deg2rad(180)  # 180 degrees east

    result = utc2solar_hour(hour, minute, second, lon)
    expected_result = 23 + 59 / 60.0 + 59 / 3600.0 + 180 / 15.0
    assert np.isclose(result, expected_result)

    hour = 0
    minute = 0
    second = 0
    lon = np.deg2rad(-180)  # 180 degrees west

    result = utc2solar_hour(hour, minute, second, lon)
    expected_result = 0 + 0 / 60.0 + 0 / 3600.0 - 180 / 15.0
    assert np.isclose(result, expected_result)


def test_realistic_cases():
    # Test edge cases
    hour = 23 * np.ones(3)
    minute = 59 * np.ones(3)
    second = 59 * np.ones(3)
    lon = np.deg2rad(np.array([2.33472, 7.75, -4.48]))  # Paris, Strasbourg, Brest

    result = utc2solar_hour(hour, minute, second, lon)
    expected_result = 23 + 59 / 60.0 + 59 / 3600.0

    assert result[0] > expected_result
    assert result[1] > result[0]
    assert result[2] < expected_result
