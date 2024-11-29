# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.power import PowerTerm


# Tests for PowerTerm
def test_power_term_value_with_scalar():
    power_term = PowerTerm()
    T = 25.0

    result = power_term.value(T)

    assert result == 0.0


def test_power_term_value_with_array():
    power_term = PowerTerm()
    T = np.array([25.0, 30.0, 35.0])
    expected = np.array([0.0, 0.0, 0.0])

    result = power_term.value(T)

    np.testing.assert_array_equal(result, expected)


def test_power_term_derivative_with_scalar():
    power_term = PowerTerm()
    temperature = 25.0

    result = power_term.derivative(temperature)

    assert result == 0.0


def test_power_term_derivative_with_array():
    power_term = PowerTerm()
    temperatures = np.array([25.0, 30.0, 35.0])
    expected = np.array([0.0, 0.0, 0.0])

    result = power_term.derivative(temperatures)

    np.testing.assert_array_equal(result, expected)
