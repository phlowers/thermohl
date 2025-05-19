# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from thermohl.power.rte import ConvectiveCooling

conv_cool_instances = [
    ConvectiveCooling(
        alt=np.array([100.0]),
        azm=np.array([2]),
        Ta=np.array([25.0]),
        ws=np.array([10.0]),
        wa=np.array([11.0]),
        D=np.array([0.01]),
        alpha=np.array([0.5]),
    ),
    ConvectiveCooling(
        alt=100.0,
        azm=2,
        Ta=25.0,
        ws=10.0,
        wa=11.0,
        D=0.01,
        alpha=0.5,
    ),
]


@pytest.mark.parametrize(
    "convective_cooling",
    conv_cool_instances,
    ids=["ConvectiveCooling with arrays", "ConvectiveCooling with scalars"],
)
def test_convective_cooling_value(convective_cooling):
    temperature = np.array([30.0])
    expected_value = 9.50218

    result = convective_cooling.value(temperature)

    np.testing.assert_allclose(result, expected_value, rtol=1e-5)
