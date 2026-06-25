# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from thermohl.utils import bisect_v, quasi_newton_2d

_nprs = 3141592654


def test_bisect():
    np.random.seed(_nprs)
    size = 99
    tol = 1.0e-09

    c = np.linspace(1, size, size) + np.random.randn(size)
    c = np.abs(c)

    def fun(x):
        return x**2 - c

    r0 = np.sqrt(c)

    x0, err = bisect_v(
        fun,
        0.0,
        np.sqrt(np.max(c)) * 1.1,
        (size,),
        print_error=False,
        tolerance=tol,
        max_iterations=99,
    )

    assert np.max(np.abs(x0 - r0) <= np.minimum(tol, err))


def test_bisect_scalar():
    def f(x):
        return x**2 - 2

    tol = 1e-6
    x0, err = bisect_v(f, lower_bound=0, upper_bound=2, output_shape=1, tolerance=tol)
    np.testing.assert_allclose(x0, np.sqrt(2), atol=tol)


def test_bisect_vector():
    def f(x):
        return np.array([x[0] ** 2 - 2, x[1] ** 3 - 2])
        # Not the best way to use bisect_v, but just for testing.

    tol = 1e-6
    x0, err = bisect_v(
        f, lower_bound=0, upper_bound=2, output_shape=(2,), tolerance=tol
    )
    np.testing.assert_allclose(x0, np.array([np.sqrt(2), np.cbrt(2)]), atol=tol)


def test_bisect_array():
    c = np.arange(1, 28).reshape(3, 3, 3)

    def f(x):
        return x**2 - c

    tol = 1e-6
    x0, err = bisect_v(
        f, lower_bound=0, upper_bound=30, output_shape=c.shape, tolerance=tol
    )

    np.testing.assert_allclose(x0, np.sqrt(c), atol=tol)


def test_bisect_no_convergence():
    def f(x):
        return x**2 + 1  # No root

    x0, err = bisect_v(
        f,
        lower_bound=-50,
        upper_bound=50,
        output_shape=(1,),
    )
    # f(x) = x^2 + 1. f(-50) = 2501 > 0.
    # Should return lower_bound = -50.0
    assert x0[0] == -50.0


def test_bisect_no_convergence_array():
    # c = 1 -> x^2 - 1 = 0, root at 1.0 (valid in [0, 2])
    # c = 3 -> x^2 + 1 = 0, no root (f(0)=1, f(2)=5) (invalid in [0, 2])
    c = np.array([1.0, 3.0])

    def f(x):
        return x**2 - 2.0 + c

    x0, err = bisect_v(
        f,
        lower_bound=0.0,
        upper_bound=2.0,
        output_shape=(2,),
    )
    assert np.allclose(x0, [1, 0])


def test_bisect_all_convergent():
    c = np.array([1.0, 4.0, 9.0])

    def f(x):
        return x**2 - c

    tol = 1e-6
    x0, err = bisect_v(f, 0.0, 10.0, output_shape=(3,), tolerance=tol)
    np.testing.assert_allclose(x0, np.sqrt(c), atol=tol)


def test_bisect_all_non_convergent():
    def f(x):
        return x**2 + 1.0  # f(x) > 0 always

    x0, err = bisect_v(f, 0.0, 10.0, output_shape=(3,))
    # f(0) = 1 > 0, so returns lower_bound = 0.0
    np.testing.assert_allclose(x0, np.zeros(3))


def test_bisect_mixed_convergence():
    # c = -1 -> no root (x^2 + 1 = 0)
    # c = 1 -> root at 1 (x^2 - 1 = 0)
    # c = 4 -> root at 2 (x^2 - 4 = 0)
    c = np.array([-1.0, 1.0, 4.0])

    def f(x):
        return x**2 - c

    tol = 1e-6
    x0, err = bisect_v(f, 0.0, 10.0, output_shape=(3,), tolerance=tol)

    # First case should be lower_bound = 0.0 because f(0) = 0^2 - (-1) = 1 > 0
    assert x0[0] == 0.0
    # Other cases should have converged
    np.testing.assert_allclose(x0[1:], [1.0, 2.0], atol=tol)


#
def test_quasi_newton_2d_convergence():
    np.random.seed(_nprs)
    size = 10
    tol = 1.0e-12

    a = np.abs(1 + np.random.randn(size))
    b = np.abs(1 + np.random.randn(size))

    def f(x, y):
        return y - a * x**2, y - b * x**3

    xg = np.ones((size,))
    yg = np.ones((size,))
    x, y, count, err = quasi_newton_2d(
        f,
        x_init=xg,
        y_init=yg,
        relative_tolerance=tol,
        max_iterations=999,
        delta_x=1.0e-09,
        delta_y=1.0e-09,
    )

    f1, f2 = f(x, y)
    assert np.logical_and(np.max(np.abs(f1)) < tol, np.max(np.abs(f2)) < tol)
    assert count == 999


def test_quasi_newton_2d_no_convergence():
    np.random.seed(_nprs)
    size = 10
    tol = 1.0e-12

    a = np.abs(1 + np.random.randn(size))
    b = np.abs(1 + np.random.randn(size))

    def f(x, y):
        return y - a * x**2, y - b * x**3

    xg = np.ones((size,))
    yg = np.ones((size,))
    x, y, count, err = quasi_newton_2d(
        f,
        x_init=xg,
        y_init=yg,
        relative_tolerance=tol,
        max_iterations=1,  # Set max iterations to 1 to force no convergence
        delta_x=1.0e-09,
        delta_y=1.0e-09,
    )

    f1, f2 = f(x, y)
    assert count == 1
    assert np.logical_or(np.max(np.abs(f1)) >= tol, np.max(np.abs(f2)) >= tol)


def test_quasi_newton_2d_large_system():
    np.random.seed(_nprs)
    size = 1000
    tol = 1.0e-12

    a = np.abs(1 + np.random.randn(size))
    b = np.abs(1 + np.random.randn(size))

    def f(x, y):
        return y - a * x**2, y - b * x**3

    xg = np.ones((size,))
    yg = np.ones((size,))
    x, y, count, err = quasi_newton_2d(
        f,
        x_init=xg,
        y_init=yg,
        relative_tolerance=tol,
        max_iterations=999,
        delta_x=1.0e-09,
        delta_y=1.0e-09,
    )

    f1, f2 = f(x, y)
    assert np.logical_and(np.max(np.abs(f1)) < tol, np.max(np.abs(f2)) < tol)
