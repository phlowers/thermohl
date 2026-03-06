# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timezone
import numpy as np
from numpy import ndarray

from thermohl.solver.base import Args, reshape, _set_dates


# Tests class Args
def test_max_len_with_mixed_types():
    dic = {
        "latitude": np.array([45.0, 46.0]),
        "longitude": 10.0,
        "altitude": np.array([20.0, 25.0, 30.0]),
    }
    args = Args(dic)

    result = args.get_number_of_computations()

    assert result == 3


def test_max_len_with_ndarray():
    dic = {
        "latitude": np.array([45.0, 46.0]),
        "longitude": np.array([10.0, 11.0]),
    }
    args = Args(dic)

    result = args.get_number_of_computations()

    assert result == 2


def test_max_len_with_scalar():
    dic = {"latitude": 45.0, "longitude": 10.0}
    args = Args(dic)

    result = args.get_number_of_computations()

    assert result == 1


def test_max_len_with_empty_dict():
    args = Args({})

    result = args.get_number_of_computations()

    assert result == 1


def test_max_len_with_varied_lengths():
    dic = {
        "latitude": np.array([45.0, 46.0]),
        "longitude": np.array([10.0]),
        "altitude": np.array([20.0, 25.0, 30.0]),
    }
    args = Args(dic)

    result = args.get_number_of_computations()

    assert result == 3


def test_extend_to_max_len_with_nd_array():
    dic = {"latitude": np.array([45.0, 46.0]), "longitude": 10.0}
    args = Args(dic)

    args.extend()

    assert isinstance(args.latitude, ndarray)
    assert isinstance(args.longitude, ndarray)
    assert len(args.latitude) == 2
    assert len(args.longitude) == 2
    np.testing.assert_array_equal(args.latitude, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.longitude, np.array([10.0, 10.0]))


def test_extend_to_max_len_with_scalar():
    dic = {"latitude": 45.0, "longitude": 10.0}
    args = Args(dic)

    args.extend()

    assert isinstance(args.latitude, ndarray)
    assert isinstance(args.longitude, ndarray)
    assert len(args.latitude) == 1
    assert len(args.longitude) == 1
    np.testing.assert_array_equal(args.latitude, np.array([45.0]))
    np.testing.assert_array_equal(args.longitude, np.array([10.0]))


def test_extend_to_max_len_with_mixed_types():
    dic = {
        "latitude": np.array([45.0, 46.0]),
        "longitude": 10.0,
        "altitude": 20.0,
    }
    args = Args(dic)

    args.extend()

    assert isinstance(args.latitude, ndarray)
    assert isinstance(args.longitude, ndarray)
    assert isinstance(args.altitude, ndarray)
    assert len(args.latitude) == 2
    assert len(args.longitude) == 2
    assert len(args.altitude) == 2
    np.testing.assert_array_equal(args.latitude, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.longitude, np.array([10.0, 10.0]))
    np.testing.assert_array_equal(args.altitude, np.array([20.0, 20.0]))


def test_extend_to_max_len_with_empty_dict():
    args = Args({})

    args.extend()

    for key in args.keys():
        assert isinstance(args[key], (float, int, ndarray, datetime))
        if isinstance(args[key], ndarray):
            assert len(args[key]) == 1


def test_compress_with_unique_values():
    dic = {
        "latitude": np.array([45.0, 45.0]),
        "longitude": np.array([10.0, 10.0]),
    }
    args = Args(dic)

    args.compress()

    assert isinstance(args.latitude, float)
    assert args.latitude == 45.0
    assert isinstance(args.longitude, float)
    assert args.longitude == 10.0


def test_compress_with_non_unique_values():
    dic = {
        "latitude": np.array([45.0, 46.0]),
        "longitude": np.array([10.0, 11.0]),
    }
    args = Args(dic)

    args.compress()

    assert isinstance(args.latitude, ndarray)
    assert isinstance(args.longitude, ndarray)
    np.testing.assert_array_equal(args.latitude, np.array([45.0, 46.0]))
    np.testing.assert_array_equal(args.longitude, np.array([10.0, 11.0]))


def test_compress_with_mixed_values():
    dic = {
        "latitude": np.array([45.0, 45.0]),
        "longitude": np.array([10.0, 11.0]),
        "altitude": 20.0,
    }
    args = Args(dic)

    args.compress()

    assert isinstance(args.latitude, float)
    assert args.latitude == 45.0
    assert isinstance(args.longitude, ndarray)
    assert isinstance(args.altitude, float)
    np.testing.assert_array_equal(args.longitude, np.array([10.0, 11.0]))
    assert args.altitude == 20.0


def test_compress_with_empty_dict():
    args = Args({})
    args.compress()
    for key in args.keys():
        assert isinstance(args[key], (float, np.int64, ndarray, datetime))


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


def test_reshape_invalid_shape():
    array = np.array(0)  # ([1.0, 2.0, 3.0])
    nb_row = 2
    nb_columns = 2
    expected = np.array([[0, 0], [0, 0]])

    result = reshape(array, nb_row, nb_columns)

    np.testing.assert_array_equal(result, expected)


def test_set_dates_single_day():
    datetime_utc = datetime(2000, 1, 1, 0, tzinfo=timezone.utc)
    offset = np.array([0, 3600, 7200])
    n = 1

    result = _set_dates(datetime_utc, offset, n)

    assert result.shape == (3, 1)
    assert result[0, 0] == datetime(2000, 1, 1, 0, tzinfo=timezone.utc)
    assert result[1, 0] == datetime(2000, 1, 1, 1, tzinfo=timezone.utc)
    assert result[2, 0] == datetime(2000, 1, 1, 2, tzinfo=timezone.utc)


def test_set_dates_multiple_days():
    datetime_utc = datetime(2000, 1, 1, 23, tzinfo=timezone.utc)
    offset = np.array([0, 3600, 7200])
    n = 1

    result = _set_dates(datetime_utc, offset, n)

    assert result.shape == (3, 1)
    assert result[0, 0] == datetime(2000, 1, 1, 23, tzinfo=timezone.utc)
    assert result[1, 0] == datetime(2000, 1, 2, 0, tzinfo=timezone.utc)
    assert result[2, 0] == datetime(2000, 1, 2, 1, tzinfo=timezone.utc)


def test_set_dates_multiple_months():
    datetime_utc = datetime(2000, 12, 31, 23, tzinfo=timezone.utc)
    offset = np.array([0, 3600, 7200])
    n = 1

    result = _set_dates(datetime_utc, offset, n)

    assert result.shape == (3, 1)
    assert result[0, 0] == datetime(2000, 12, 31, 23, tzinfo=timezone.utc)
    assert result[1, 0] == datetime(2001, 1, 1, 0, tzinfo=timezone.utc)
    assert result[2, 0] == datetime(2001, 1, 1, 1, tzinfo=timezone.utc)


def test_set_dates_multiple_inputs():
    datetime_utc = [
        datetime(2000, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2000, 2, 2, 12, tzinfo=timezone.utc),
    ]
    offset = np.array([0, 3600, 7200])
    n = 2

    result = _set_dates(datetime_utc, offset, n)

    assert result.shape == (3, 2)

    assert result[0, 0] == datetime(2000, 1, 1, 0, tzinfo=timezone.utc)
    assert result[1, 0] == datetime(2000, 1, 1, 1, tzinfo=timezone.utc)
    assert result[2, 0] == datetime(2000, 1, 1, 2, tzinfo=timezone.utc)

    assert result[0, 1] == datetime(2000, 2, 2, 12, tzinfo=timezone.utc)
    assert result[1, 1] == datetime(2000, 2, 2, 13, tzinfo=timezone.utc)
    assert result[2, 1] == datetime(2000, 2, 2, 14, tzinfo=timezone.utc)
