# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from numpy import ndarray

from thermohl.solver.base import Args, reshape, _set_dates


# Tests class Args
def test_max_len_with_mixed_types():
    dic = {
        "lat": np.array([45.0, 46.0]),
        "lon": 10.0,
        "alt": np.array([20.0, 25.0]),
    }
    args = Args(dic)

    result = args.shape()

    assert result == (2,)


def test_shape_with_ndarray():
    dic = {"lat": np.array([45.0, 46.0]), "lon": np.array([10.0, 11.0])}
    args = Args(dic)

    result = args.shape()

    assert result == (2,)


def test_shape_with_scalar():
    dic = {"lat": 45.0, "lon": 10.0}
    args = Args(dic)

    result = args.shape()

    assert result == ()


def test_shape_with_empty_dict():
    args = Args({})

    result = args.shape()

    assert result == ()


def test_shape_with_varied_lengths():
    dic = {
        "lat": np.array([45.0, 46.0]),
        "lon": np.array([10.0]),
        "alt": np.array([20.0, 25.0, 30.0]),
    }
    try:
        args = Args(dic)
        assert False
    except ValueError:
        pass


def test_extend_with_nd_array():
    dic = {"lat": np.array([45.0, 46.0]), "lon": 10.0}
    args = Args(dic)

    args.extend()

    assert isinstance(args.lat, ndarray)
    assert isinstance(args.lon, ndarray)
    assert len(args.lat) == 2
    assert len(args.lon) == 2
    np.testing.assert_array_equal(args.lat, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.lon, np.array([10.0, 10.0]))


def test_extend_with_scalar():
    dic = {"lat": 45.0, "lon": 10.0}
    args = Args(dic)

    args.extend()

    assert isinstance(args.lat, ndarray)
    assert isinstance(args.lon, ndarray)
    assert len(args.lat) == 1
    assert len(args.lon) == 1
    np.testing.assert_array_equal(args.lat, np.array([45.0]))
    np.testing.assert_array_equal(args.lon, np.array([10.0]))


def test_extend_with_mixed_types():
    dic = {"lat": np.array([45.0, 46.0]), "lon": 10.0, "alt": np.array(20.0)}
    args = Args(dic)

    args.extend()

    assert isinstance(args.lat, ndarray)
    assert isinstance(args.lon, ndarray)
    assert isinstance(args.alt, ndarray)
    assert len(args.lat) == 2
    assert len(args.lon) == 2
    assert len(args.alt) == 2
    np.testing.assert_array_equal(args.lat, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.lon, np.array([10.0, 10.0]))
    np.testing.assert_array_equal(args.alt, np.array([20.0, 20.0]))


def test_extend_with_empty_dict():
    args = Args({})

    args.extend()

    for key in args.keys():
        assert isinstance(args[key], (float, int, ndarray))
        if isinstance(args[key], ndarray):
            assert len(args[key]) == 1


def test_compress_with_unique_values():
    dic = {"lat": np.array([45.0, 45.0]), "lon": np.array([10.0, 10.0])}
    args = Args(dic)

    args.compress()

    assert isinstance(args.lat, float)
    assert args.lat == 45.0
    assert isinstance(args.lon, float)
    assert args.lon == 10.0


def test_compress_with_non_unique_values():
    dic = {"lat": np.array([45.0, 46.0]), "lon": np.array([10.0, 11.0])}
    args = Args(dic)

    args.compress()

    assert isinstance(args.lat, ndarray)
    assert isinstance(args.lon, ndarray)
    np.testing.assert_array_equal(args.lat, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.lon, np.array([10.0, 11.0]))


def test_compress_with_mixed_values():
    dic = {"lat": np.array([45.0, 45.0]), "lon": np.array([10.0, 11.0]), "alt": 20.0}
    args = Args(dic)

    args.compress()

    assert isinstance(args.lat, float)
    assert args.lat == 45.0
    assert isinstance(args.lon, ndarray)
    assert isinstance(args.alt, float)
    np.testing.assert_array_equal(args.lon, np.array([10.0, 11.0]))
    assert args.alt == 20.0


def test_compress_with_empty_dict():
    args = Args({})

    args.compress()

    for key in args.keys():
        assert isinstance(args[key], (float, int, np.integer, np.floating, ndarray))
        if isinstance(args[key], ndarray):
            assert len(args[key]) == 1


# Tests Fonctions Base
def test_reshape_on_one_row_array():
    array = np.array([1.0, 2.0, 3.0])
    nb_row = 3
    nb_columns = 1
    expected = np.array([[1.0], [2.0], [3.0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_1d_to_2d_column_stack():
    array = np.array([1.0, 2.0, 3.0])
    nb_row = 3
    nb_columns = 1
    expected = np.array([[1.0], [2.0], [3.0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_1d_to_2d_row_stack():
    array = np.array([1.0, 2.0, 3.0])
    nb_row = 1
    nb_columns = 3
    expected = np.array([[1.0, 2.0, 3.0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_2d_to_2d():
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nb_row = 2
    nb_columns = 3
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_scalar_to_2d():
    scalar = 1.0
    nb_row = 2
    nb_columns = 3
    expected = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    result = reshape(scalar, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_another_scalar_to_2d():
    array = np.array(0)
    nb_row = 2
    nb_columns = 2
    expected = np.array([[0, 0], [0, 0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_reshape_invalid_shape():
    array = np.array([1.0, 2.0, 3.0])
    nb_row = 2
    nb_columns = 2
    try:
        reshape(array, nb_row, nb_columns)
        assert False
    except ValueError:
        pass


def test_set_dates_single_day():
    month = np.array([1])
    day = np.array([1])
    hour = np.array([0])
    t = np.array([0, 3600, 7200])
    n = 1

    months, days, hours = _set_dates(month, day, hour, t, n)

    assert months.shape == (3, 1)
    assert days.shape == (3, 1)
    assert hours.shape == (3, 1)
    assert months[0, 0] == 1
    assert days[0, 0] == 1
    assert hours[0, 0] == 0.0
    assert months[1, 0] == 1
    assert days[1, 0] == 1
    assert hours[1, 0] == 1.0
    assert months[2, 0] == 1
    assert days[2, 0] == 1
    assert hours[2, 0] == 2.0


def test_set_dates_multiple_days():
    month = np.array([1])
    day = np.array([1])
    hour = np.array([23])
    t = np.array([0, 3600, 7200])
    n = 1

    months, days, hours = _set_dates(month, day, hour, t, n)

    assert months.shape == (3, 1)
    assert days.shape == (3, 1)
    assert hours.shape == (3, 1)
    assert months[0, 0] == 1
    assert days[0, 0] == 1
    assert hours[0, 0] == 23.0
    assert months[1, 0] == 1
    assert days[1, 0] == 2
    assert hours[1, 0] == 0.0
    assert months[2, 0] == 1
    assert days[2, 0] == 2
    assert hours[2, 0] == 1.0


def test_set_dates_multiple_months():
    month = np.array([12])
    day = np.array([31])
    hour = np.array([23])
    t = np.array([0, 3600, 7200])
    n = 1

    months, days, hours = _set_dates(month, day, hour, t, n)

    assert months.shape == (3, 1)
    assert days.shape == (3, 1)
    assert hours.shape == (3, 1)
    assert months[0, 0] == 12
    assert days[0, 0] == 31
    assert hours[0, 0] == 23.0
    assert months[1, 0] == 1
    assert days[1, 0] == 1
    assert hours[1, 0] == 0.0
    assert months[2, 0] == 1
    assert days[2, 0] == 1
    assert hours[2, 0] == 1.0


def test_set_dates_multiple_inputs():
    month = np.array([1, 2])
    day = np.array([1, 2])
    hour = np.array([0, 12])
    t = np.array([0, 3600, 7200])
    n = 2

    months, days, hours = _set_dates(month, day, hour, t, n)

    assert months.shape == (3, 2)
    assert days.shape == (3, 2)
    assert hours.shape == (3, 2)

    assert months[0, 0] == 1
    assert days[0, 0] == 1
    assert hours[0, 0] == 0.0

    assert months[1, 0] == 1
    assert days[1, 0] == 1
    assert hours[1, 0] == 1.0

    assert months[2, 0] == 1
    assert days[2, 0] == 1
    assert hours[2, 0] == 2.0

    assert months[0, 1] == 2
    assert days[0, 1] == 2
    assert hours[0, 1] == 12.0

    assert months[1, 1] == 2
    assert days[1, 1] == 2
    assert hours[1, 1] == 13.0

    assert months[2, 1] == 2
    assert days[2, 1] == 2
    assert hours[2, 1] == 14.0
