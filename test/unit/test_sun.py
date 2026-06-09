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
    datetime_utc = np.datetime64("2025-05-18T12:30:45")
    longitude = np.deg2rad(45)  # 45 degrees east

    result = utc2solar_hour(datetime_utc, longitude)
    expected_result = 15.57397066555581
    assert np.isclose(result, expected_result)


def test_array_input():
    # Test with numpy array inputs
    datetimes_utc = np.array(
        [
            np.datetime64("2025-03-18T12:30:00"),
            np.datetime64("2025-09-02T15:45:00"),
            np.datetime64("2025-12-27T18:00:00"),
        ]
    )
    longitudes = np.deg2rad(np.array([0, 90, 135]))

    result = utc2solar_hour(datetimes_utc, longitudes)
    expected_result = np.array([12.35394, 21.76352, 26.976276])
    np.testing.assert_array_almost_equal(result, expected_result)
