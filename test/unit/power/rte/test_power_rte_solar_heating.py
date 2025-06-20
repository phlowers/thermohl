# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from thermohl.power.rte.solar_heating import solar_irradiance


def test_solar_irradiance_positive_altitude():
    """Test solar irradiance with positive solar altitude."""
    lat = np.deg2rad(45.0)  # Latitude in radians
    month = 6  # June
    day = 21  # Summer solstice
    hour = 12.0  # Noon

    result = solar_irradiance(lat, month, day, hour)

    assert result > 0.0


def test_solar_irradiance_negative_altitude():
    """Test solar irradiance with negative solar altitude."""
    lat = np.deg2rad(45.0)  # Latitude in radians
    month = 12  # December
    day = 21  # Winter solstice
    hour = 0.0  # Midnight

    result = solar_irradiance(lat, month, day, hour)

    assert result == 0.0


def test_solar_irradiance_array_input():
    """Test solar irradiance with array inputs."""
    lat = np.array([np.deg2rad(45.0), np.deg2rad(55.0)])
    month = np.array([6, 12])
    day = np.array([21, 21])
    hour = np.array([12.0, 0.0])

    result = solar_irradiance(lat, month, day, hour)

    assert result.shape == (2,)
    assert result[0] > 0.0  # Positive solar altitude
    assert result[1] == 0.0  # Negative solar altitude
