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
    hour = 12 + 30 / 60 + 45 / 3600
    day = 18
    month = 5
    longitude = np.deg2rad(45)  # 45 degrees east

    result = utc2solar_hour(hour, day, month, longitude)
    expected_result = 15.57397066555581
    assert np.isclose(result, expected_result)


def test_array_input():
    # Test with numpy array inputs
    hours = np.array([12, 15, 18]) + np.array([30, 45, 0]) / 60
    days = np.array([18, 2, 27])
    months = np.array([3, 9, 12])
    lons = np.deg2rad(np.array([0, 90, 135]))  # 45, 90, and 135 degrees east

    result = utc2solar_hour(hours, days, months, lons)
    expected_result = np.array([12.35394, 21.76352, 26.976276])
    np.testing.assert_array_almost_equal(result, expected_result)
