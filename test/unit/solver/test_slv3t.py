# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import nan
from unittest.mock import MagicMock

import numpy as np

from thermohl.solver import Solver3T
from thermohl.solver.slv3t import (
    _profile_mom,
    _phi,
    _profile_bim_avg_coeffs,
)


def test_profile_mom_basic():
    ts = np.array([300])
    tc = np.array([400])
    r = np.array([0.5])
    re = np.array([1.0])
    expected = np.array([375])

    result = _profile_mom(ts, tc, r, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_profile_mom_multiple_values():
    ts = np.array([300, 350])
    tc = np.array([400, 450])
    r = np.array([0.5, 0.2])
    re = np.array([1.0, 1.0])
    expected = np.array([375, 446])

    result = _profile_mom(ts, tc, r, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_profile_mom_edge_case_zero_radius():
    ts = np.array([300])
    tc = np.array([400])
    r = np.array([0.0])
    re = np.array([1.0])
    expected = np.array([400])

    result = _profile_mom(ts, tc, r, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_profile_mom_edge_case_equal_temperatures():
    ts = np.array([300])
    tc = np.array([300])
    r = np.array([0.5])
    re = np.array([1.0])
    expected = np.array([300])

    result = _profile_mom(ts, tc, r, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_profile_mom_large_values():
    ts = np.array([1e6])
    tc = np.array([2e6])
    r = np.array([0.5])
    re = np.array([1.0])
    expected = np.array([1.75e6])

    result = _profile_mom(ts, tc, r, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_phi_basic():
    r = 0.6
    ri = 0.5
    re = 1.0
    expected = 0.012559

    result = _phi(r, ri, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_phi_multiple_values():
    r = np.array([0.5, 0.8])
    ri = np.array([0.2, 0.3])
    re = np.array([1.0, 1.0])
    expected = np.array([0.071196, 0.205193])

    result = _phi(r, ri, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_phi_edge_case_zero_radius():
    r = np.array([0.2])
    ri = np.array([0.2])
    re = np.array([1.0])
    expected = np.array([0.0])

    result = _phi(r, ri, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_phi_edge_case_equal_radii():
    r = np.array([0.5])
    ri = np.array([0.5])
    re = np.array([1.0])
    expected = np.array([0.0])

    result = _phi(r, ri, re)

    np.testing.assert_array_almost_equal(result, expected)


def test_phi_large_values():
    r = np.array([1e6])
    ri = np.array([1e5])
    re = np.array([1e7])
    expected = np.array([0.00472])

    result = _phi(r, ri, re)

    np.testing.assert_array_almost_equal(result, expected)


# def test_profile_bim_basic():
#     ts = np.array([300])
#     tc = np.array([400])
#     r = np.array([0.5])
#     ri = np.array([0.2])
#     re = np.array([1.0])
#     expected = np.array([383.55518])
#
#     result = _profile_bim(ts, tc, r, ri, re)
#
#     np.testing.assert_array_almost_equal(result, expected)
#
#
# def test_profile_bim_multiple_values():
#     ts = np.array([300, 350])
#     tc = np.array([400, 450])
#     r = np.array([0.5, 0.2])
#     ri = np.array([0.2, 0.3])
#     re = np.array([1.0, 1.0])
#     expected = np.array([325, 350])
#
#     result = _profile_bim(ts, tc, r, ri, re)
#
#     np.testing.assert_array_almost_equal(result, expected)
#
#
# def test_profile_bim_edge_case_zero_radius():
#     ts = np.array([300])
#     tc = np.array([400])
#     r = np.array([0.0])
#     ri = np.array([0.2])
#     re = np.array([1.0])
#     expected = np.array([400])
#
#     result = _profile_bim(ts, tc, r, ri, re)
#
#     np.testing.assert_array_almost_equal(result, expected)
#
#
# def test_profile_bim_edge_case_equal_temperatures():
#     ts = np.array([300])
#     tc = np.array([300])
#     r = np.array([0.5])
#     ri = np.array([0.2])
#     re = np.array([1.0])
#     expected = np.array([300])
#
#     result = _profile_bim(ts, tc, r, ri, re)
#
#     np.testing.assert_array_almost_equal(result, expected)
#
#
# def test_profile_bim_large_values():
#     ts = np.array([1e6])
#     tc = np.array([2e6])
#     r = np.array([0.5])
#     ri = np.array([0.2])
#     re = np.array([1.0])
#     expected = np.array([1835551.796551])
#
#     result = _profile_bim(ts, tc, r, ri, re)
#
#     np.testing.assert_array_almost_equal(result, expected)


def test_profile_bim_avg_coeffs_basic():
    ri = np.array([0.2])
    re = np.array([1.0])
    expected_a = np.array([0.370445])
    expected_b = np.array([0.831245])

    result_a, result_b = _profile_bim_avg_coeffs(ri, re)

    np.testing.assert_array_almost_equal(result_a, expected_a)
    np.testing.assert_array_almost_equal(result_b, expected_b)


def test_profile_bim_avg_coeffs_multiple_values():
    ri = np.array([0.2, 0.3])
    re = np.array([1.0, 1.0])
    expected_a = np.array([0.370445, 0.279235])
    expected_b = np.array([0.831245, 0.693285])

    result_a, result_b = _profile_bim_avg_coeffs(ri, re)

    np.testing.assert_array_almost_equal(result_a, expected_a)
    np.testing.assert_array_almost_equal(result_b, expected_b)


# Tests Solver3T
def test_morgan_coefficients_basic():
    solver = Solver3T()
    solver.args.D = np.array([1.0])
    solver.args.d = np.array([0.5])
    solver.update()

    expected_c = np.array([0.268951])
    expected_D = np.array([1.0])
    expected_d = np.array([0.5])
    expected_i = np.array([0])

    result_c, result_D, result_d, result_i = solver._morgan_coefficients()

    np.testing.assert_array_almost_equal(result_c, expected_c)
    np.testing.assert_array_almost_equal(result_D, expected_D)
    np.testing.assert_array_almost_equal(result_d, expected_d)
    np.testing.assert_array_equal(result_i, expected_i)


def test_morgan_coefficients_multiple_values():
    solver = Solver3T()
    solver.args.D = np.array([1.0, 2.5])
    solver.args.d = np.array([0.5, 1.0])
    solver.update()

    expected_c = np.array([0.268951, 0.325468])
    expected_D = np.array([1.0, 2.5])
    expected_d = np.array([0.5, 1.0])
    expected_i = np.array([0, 1])

    result_c, result_D, result_d, result_i = solver._morgan_coefficients()

    np.testing.assert_array_almost_equal(result_c, expected_c)
    np.testing.assert_array_almost_equal(result_D, expected_D)
    np.testing.assert_array_almost_equal(result_d, expected_d)
    np.testing.assert_array_equal(result_i, expected_i)


def test_morgan_coefficients_edge_case_zero_d():
    solver = Solver3T()
    solver.args.D = np.array([1.0])
    solver.args.d = np.array([0.0])
    solver.update()

    expected_c = np.array([0.5])
    expected_D = np.array([1.0])
    expected_d = np.array([0.0])
    expected_i = np.array([])

    result_c, result_D, result_d, result_i = solver._morgan_coefficients()

    np.testing.assert_array_almost_equal(result_c, expected_c)
    np.testing.assert_array_almost_equal(result_D, expected_D)
    np.testing.assert_array_almost_equal(result_d, expected_d)
    np.testing.assert_array_equal(result_i, expected_i)


def test_morgan_coefficients_edge_case_equal_D_and_d():
    solver = Solver3T()
    solver.args.D = np.array([1.0])
    solver.args.d = np.array([1.0])
    solver.update()

    expected_c = np.array([nan])
    expected_D = np.array([1.0])
    expected_d = np.array([1.0])
    expected_i = np.array([0])

    result_c, result_D, result_d, result_i = solver._morgan_coefficients()

    np.testing.assert_array_almost_equal(result_c, expected_c)
    np.testing.assert_array_almost_equal(result_D, expected_D)
    np.testing.assert_array_almost_equal(result_d, expected_d)
    np.testing.assert_array_equal(result_i, expected_i)


def test_morgan_coefficients_large_values():
    solver = Solver3T()
    solver.args.D = np.array([1e6])
    solver.args.d = np.array([1e5])
    solver.update()

    expected_c = np.array([0.476742])
    expected_D = np.array([1e6])
    expected_d = np.array([1e5])
    expected_i = np.array([0])

    result_c, result_D, result_d, result_i = solver._morgan_coefficients()

    np.testing.assert_array_almost_equal(result_c, expected_c)
    np.testing.assert_array_almost_equal(result_D, expected_D)
    np.testing.assert_array_almost_equal(result_d, expected_d)
    np.testing.assert_array_equal(result_i, expected_i)


def test_joule_basic():
    solver = Solver3T()
    solver.mgc = (np.array([0.5]), np.array([1.0]), np.array([0.5]), np.array([0]))
    ts = np.array([300.0])
    tc = np.array([400.0])
    expected = np.array([0.0])

    result = solver.joule(ts, tc)

    np.testing.assert_array_almost_equal(result, expected)


def test_balance_basic():
    solver = Solver3T()
    solver.joule = MagicMock(return_value=np.array([100.0]))
    solver.sh = MagicMock()
    solver.sh.value = MagicMock(return_value=np.array([50.0]))
    solver.cc = MagicMock()
    solver.cc.value = MagicMock(return_value=np.array([30.0]))
    solver.rc = MagicMock()
    solver.rc.value = MagicMock(return_value=np.array([10.0]))
    solver.pc = MagicMock()
    solver.pc.value = MagicMock(return_value=np.array([5.0]))

    ts = np.array([300])
    tc = np.array([400])
    expected = np.array([105.0])

    result = solver.balance(ts, tc)

    np.testing.assert_array_almost_equal(result, expected)
    solver.joule.assert_called_once_with(ts, tc)
    solver.sh.value.assert_called_once_with(ts)
    solver.cc.value.assert_called_once_with(ts)
    solver.rc.value.assert_called_once_with(ts)
    solver.pc.value.assert_called_once_with(ts)


def test_morgan_basic():
    solver = Solver3T()
    solver.mgc = (np.array([0.5]), np.array([1.0]), np.array([0.5]), np.array([0]))
    solver.jh = MagicMock()
    solver.jh.value = MagicMock(return_value=np.array([100.0]))
    solver.joule = MagicMock(return_value=np.array([100.0]))
    solver.args = MagicMock()
    solver.args.l = 1.0

    ts = np.array([300])
    tc = np.array([400])
    expected = np.array([92.042])

    result = solver.morgan(ts, tc)

    np.testing.assert_array_almost_equal(result, expected, decimal=3)
    solver.joule.assert_called_once_with(ts, tc)
