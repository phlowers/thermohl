# Copyright 2023 Eurobios Mews Labs
# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


"""Misc. utility code for thermohl project."""

import os

import numpy as np
import pandas as pd
import yaml


def _dict_completion(
    dat: dict, filename: str, check: bool = True, warning: bool = False
) -> dict:
    """Complete input dict with values from file.

    Read dict stored in filename (yaml format) and for each key in it, add it
    to input dict dat if the key is not already in dat.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    dfl = yaml.safe_load(open(fil, "r"))
    for k in dfl.keys():
        if k not in dat.keys() or dat[k] is None:
            dat[k] = dfl[k]
            if warning:
                print("Added key %s from default parameters" % (k,))
        elif (
            not isinstance(dat[k], int)
            and not isinstance(dat[k], float)
            and not isinstance(dat[k], np.ndarray)
            and check
        ):
            raise TypeError(
                "element in input dict (key [%s]) must be int, float or numpy.ndarray"
                % (k,)
            )
    return dat


def add_default_parameters(dat: dict, warning: bool = False) -> dict:
    """Add default parameters if there is missing input.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_values.yaml"
    )
    return _dict_completion(dat, fil, warning=warning)


def add_default_uncertainties(dat: dict, warning: bool = False) -> dict:
    """Add default uncertainty parameters if there is missing input.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_uncertainties.yaml"
    )
    return _dict_completion(dat, fil, check=False, warning=warning)


def df2dct(df: pd.DataFrame) -> dict:
    """Convert a pandas.DataFrame to a dictionary.

    Would be an equivalent to df.to_dict(orient='numpy.ndarray') if it existed.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        DESCRIPTION.
    """
    q = df.to_dict(orient="list")
    for k in q.keys():
        if len(q[k]) > 1:
            q[k] = np.array(q[k])
        else:
            q[k] = q[k][0]
    return q


def bisect_v(
    fun: callable,
    a: float,
    b: float,
    shape: tuple[int, ...],
    tol=1.0e-06,
    maxiter=128,
    print_err=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisection method to find a zero of a continuous function [a, b] -> R,
    such that f(a) < 0 < f(b).

    The method is vectorized, in the sense that it can in a single call find
    the zeros of several independent real-valued functions with the same input
    range [a, b].
    For this purpose, the `fun` argument should be a Python function taking as
    input a Numpy array of values in [a, b] and returning a Numpy array
    containing the evaluation of each function for the corresponding input.
    This is most efficient if the outputs values of all these functions are
    computed using vectorized Numpy operations as in the example below.

    Parameters
    ----------
    fun: Python function taking a Numpy array and returning a Numpy array of the same shape.
    a : lower bound of [a, b] interval.
    b : upper bound of [a, b] interval.
    shape : shape of the inputs and ouputs of `fun` and thus of the ouputs of `bisect_v`.
    tol : absolute tolerance.
    maxiter : maximum number of iterations.
    print_err : print or not max error and number of iterations at the end.

    Returns
    -------
    x: Numpy array of the same shape as the inputs and outputs of `fun`
        The zero of each function.
    err: Numpy array of the same shape as the inputs and outputs of `fun`
        The convergence error.

    Example
    -------
    >>> c = np.array([1.0, 4.0, 9.0, 16.0])
    >>> def f(x):
    ...     return x**2 - c
    >>> x0, err = bisect_v(f, a=0.0, b=10.0, shape=(4,), tol=1e-10)
    >>> x0
    array([1., 2., 3., 4.])

    """
    a_ = a * np.ones(shape)
    b_ = b * np.ones(shape)

    err = np.abs(b - a)
    count = 1
    while np.nanmax(err) > tol and count <= maxiter:
        x = 0.5 * (a_ + b_)
        y = fun(x)
        i = y < 0
        a_[i] = x[i]
        b_[~i] = x[~i]
        err = np.abs(b_ - a_)
        count += 1
    x = 0.5 * (a_ + b_)
    x[np.isnan(fun(x))] = np.nan
    if print_err:
        print(f"Bisection max err (abs) : {np.max(err):.2E}; count={count}")
    return x, err


# In agreement with Eurobios, this function has been retrieved from the pyntb library,
# in order to remove the external dependency on this library.
# In this library, this function was initially developed under the name qnewt2d_v
def quasi_newton_2d(
    f1: callable,
    f2: callable,
    x0: np.ndarray,
    y0: np.ndarray,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 64,
    delta_x: float = 1.0e-03,
    delta_y: float = 1.0e-03,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Two-dimensional quasi-Newton with arrays.

    Apply a 2D quasi newton on a large number of case at the same time, ie solve
    the system [f1(x, y), f2(x, y)] = [0., 0.] in n cases.

    Derivatives are estimated with a second-order centered estimation (ie f1 and
    f2 are evaluated four times at each iteration).

    All return values are arrays of the same size as inputs x0 and y0.

    Parameters
    ----------
    f1 : first component of a 2d function of two variables
    f2 : second component of a 2d function of two variables
    x0 : first component of the initial guess
    y0 : second component of the initial guess
    relative_tolerance : relative tolerance
    max_iterations : max number of iteration
    delta_x : delta for evaluating derivative regarding first component
    delta_y : delta for evaluating derivative regarding second component

    Returns
    -------
    x: first component of solution
    y: second component of solution
    count: number of iterations when exiting the function
    err: relative error when exiting the function

    """
    x = x0.copy()
    y = y0.copy()

    for count in range(max_iterations):
        # Evaluate functions at current x and y
        f1_value = f1(x, y)
        f2_value = f2(x, y)

        # Compute Jacobian matrix using second-order centered differences
        jacobian_11 = (f1(x + delta_x, y) - f1(x - delta_x, y)) / (2 * delta_x)
        jacobian_12 = (f1(x, y + delta_y) - f1(x, y - delta_y)) / (2 * delta_y)
        jacobian_21 = (f2(x + delta_x, y) - f2(x - delta_x, y)) / (2 * delta_x)
        jacobian_22 = (f2(x, y + delta_y) - f2(x, y - delta_y)) / (2 * delta_y)

        # Compute inverse of the Jacobian determinant
        inv_jacobian_det = 1.0 / (jacobian_11 * jacobian_22 - jacobian_12 * jacobian_21)
        err_abs_x = inv_jacobian_det * (jacobian_22 * f1_value - jacobian_12 * f2_value)
        err_abs_y = inv_jacobian_det * (jacobian_11 * f2_value - jacobian_21 * f1_value)

        x -= err_abs_x
        y -= err_abs_y

        # Check for convergence
        err = max(np.nanmax(np.abs(err_abs_x / x)), np.nanmax(np.abs(err_abs_y / y)))
        if err <= relative_tolerance:
            break

    return x, y, count + 1, np.maximum(np.abs(err_abs_x / x), np.abs(err_abs_y / y))
