# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from thermohl.power.rte.solar_heating import solar_irradiance


def test_solar_irradiance_positive_altitude():
    """Test solar irradiance with positive solar altitude."""
    result = solar_irradiance(np.deg2rad(45))
    assert result > 0.0


def test_solar_irradiance_negative_altitude():
    """Test solar irradiance with negative solar altitude."""
    result = solar_irradiance(np.deg2rad(-5))
    assert result == 0.0


def test_solar_irradiance_array_input():
    """Test solar irradiance with array inputs."""
    result = solar_irradiance(np.deg2rad([8, -1]))

    assert result.shape == (2,)
    assert result[0] > 0.0  # Positive solar altitude
    assert result[1] == 0.0  # Negative solar altitude
