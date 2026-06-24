# Copyright 2023 Eurobios Mews Labs
# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


"""Misc. utility code for thermohl project."""

import logging
import os
from functools import wraps
from importlib.util import find_spec

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def add_stderr_logger(level: int = logging.DEBUG) -> logging.StreamHandler:
    """Helper for quickly adding a StreamHandler to the logger.

    Args:
        level (int): Logging level.

    Returns:
        logging.StreamHandler: The added handler.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


def _dict_completion(
    params: dict,
    filename: str,
    validate_types: bool = True,
    warning: bool = False,
) -> dict:
    """Complete input dict with values from file.

    Read dict stored in filename (yaml format) and for each key in it, add it
    to input dict dat if the key is not already in dat.

    Args:
        params (dict): Input dict with parameters for power terms.
        warning (bool, optional): Print a message if a parameter is missing. The default is False.

    Returns:
        dict: Completed input dict if some parameters were missing.

    """
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    defaults = yaml.safe_load(open(file_path, "r"))
    for key in defaults.keys():
        if key not in params.keys() or params[key] is None:
            params[key] = defaults[key]
            if warning:
                logger.warning("Added key %s from default parameters" % (key,))
        elif (
            not isinstance(params[key], int)
            and not isinstance(params[key], float)
            and not isinstance(params[key], np.ndarray)
            and validate_types
        ):
            raise TypeError(
                "element in input dict (key [%s]) must be int, float or numpy.ndarray"
                % (key,)
            )
    return params


def add_default_parameters(params: dict, warning: bool = False) -> dict:
    """Add default parameters if there is missing input.

    Args:
        params (dict): Input dict with parameters for power terms.
        warning (bool, optional): Print a message if a parameter is missing. The default is False.

    Returns:
        dict: Completed input dict if some parameters were missing.

    """
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_values.yaml"
    )
    return _dict_completion(params, file_path, warning=warning)


def add_default_uncertainties(params: dict, warning: bool = False) -> dict:
    """Add default uncertainty parameters if there is missing input.

    Args:
        params (dict): Input dict with parameters for power terms.
        warning (bool, optional): Print a message if a parameter is missing. The default is False.

    Returns:
        dict: Completed input dict if some parameters were missing.

    """
    file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_uncertainties.yaml"
    )
    return _dict_completion(params, file_path, validate_types=False, warning=warning)


def bisect_v(
    func: callable,
    lower_bound: float,
    upper_bound: float,
    output_shape: tuple[int, ...],
    tolerance: float = 1.0e-06,
    max_iterations: int = 128,
    print_error: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisection method to find a zero of a continuous function [a, b] -> R,
    such that f(a) <= 0 <= f(b).

    The method is vectorized, in the sense that it can in a single call find
    the zeros of several independent real-valued functions with the same input
    range [a, b].
    For this purpose, the `fun` argument should be a Python function taking as
    input a Numpy array of values in [a, b] and returning a Numpy array
    containing the evaluation of each function for the corresponding input.
    This is most efficient if the outputs values of all these functions are
    computed using vectorized Numpy operations as in the example below.

    Args:
        func (Callable[[np.ndarray], np.ndarray]): Python function taking a NumPy array and returning a NumPy array of the same shape.
        lower_bound (float): Lower bound of the [a, b] interval.
        upper_bound (float): Upper bound of the [a, b] interval.
        output_shape (tuple[int, ...]): Shape of the inputs and outputs of `func` and thus of the outputs of `bisect_v`.
        tolerance (float): Absolute tolerance.
        max_iterations (int): Maximum number of iterations.
        print_error (bool): Whether to print the max absolute error and iteration count at the end.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - x: NumPy array with the zeros (same shape as `shape`).
            - err: NumPy array with the absolute convergence error (same shape as `shape`).

    Examples
    >>> c = np.array([1.0, 4.0, 9.0, 16.0])
    >>> def f(x):
    ...     return x**2 - c
    >>> x0, err = bisect_v(f, lower_bound=0.0, upper_bound=10.0, output_shape=(4,), tolerance=1e-10)
    >>> x0
    array([1., 2., 3., 4.])

    """
    lower_bounds = lower_bound * np.ones(output_shape)
    upper_bounds = upper_bound * np.ones(output_shape)

    # Check if the condition f(a) <= 0 <= f(b) is satisfied
    valid_mask = (func(lower_bounds) <= 0) & (func(upper_bounds) >= 0)

    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logger.warning(
            f"Bisection method: {invalid_count} case(s) do not satisfy convergence conditions "
            f"(f(a) <= 0 <= f(b)). Returning lower_bound or upper_bound for these cases."
        )

    abs_error = np.abs(upper_bounds - lower_bounds)
    iteration_count = 1
    # Only iterate on valid cases
    while (
        np.any(abs_error[valid_mask] > tolerance) and iteration_count <= max_iterations
    ):
        midpoint = 0.5 * (lower_bounds + upper_bounds)
        values = func(midpoint)
        lower_mask = (values < 0) & valid_mask
        upper_mask = (values >= 0) & valid_mask

        lower_bounds[lower_mask] = midpoint[lower_mask]
        upper_bounds[upper_mask] = midpoint[upper_mask]

        abs_error[valid_mask] = np.abs(
            upper_bounds[valid_mask] - lower_bounds[valid_mask]
        )
        iteration_count += 1

    midpoint = 0.5 * (lower_bounds + upper_bounds)
    # Handle non-convergent cases: return lower_bound if f(a) > 0, upper_bound if f(b) < 0
    # If both are true (shouldn't happen for monotonic), f_lower > 0 takes precedence here
    midpoint[~valid_mask] = np.where(
        func(lower_bounds)[~valid_mask] > 0,
        lower_bounds[~valid_mask],
        upper_bounds[~valid_mask],
    )
    midpoint[np.isnan(func(midpoint))] = np.nan
    if print_error:
        logger.info(
            f"Bisection max err (abs) : {np.nanmax(abs_error):.2E}; count={iteration_count}"
        )
    return midpoint, abs_error


# In agreement with Eurobios, this function has been retrieved from the pyntb library,
# in order to remove the external dependency on this library.
# In this library, this function was initially developed under the name qnewt2d_v
def quasi_newton_2d(
    func: callable,
    x_init: np.ndarray,
    y_init: np.ndarray,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 64,
    delta_x: float = 1.0e-03,
    delta_y: float = 1.0e-03,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Two-dimensional quasi-Newton with arrays.

    Apply a 2D quasi newton on a large number of case at the same time, ie solve
    the system [f1(x, y), f2(x, y)] = [0., 0.] in n cases.

    Derivatives are estimated with a second-order centered estimation (func is
    evaluated five times at each iteration).

    All return values are arrays of the same size as inputs x_init and y_init.

    Args:
        func (Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]):
            Function returning ``(f1(x, y), f2(x, y))`` in a single call.
        x_init (np.ndarray): First component of the initial guess.
        y_init (np.ndarray): Second component of the initial guess.
        relative_tolerance (float): Relative tolerance for convergence.
        max_iterations (int): Maximum number of iterations.
        delta_x (float): Delta for evaluating the derivative with respect to the first component.
        delta_y (float): Delta for evaluating the derivative with respect to the second component.

    Returns:
        tuple[np.ndarray, np.ndarray, int, np.ndarray]:
            - x: First component of the solution.
            - y: Second component of the solution.
            - count: Number of iterations when exiting the function.
            - err: Relative error when exiting the function (per component).

    """
    x = x_init.copy()
    y = y_init.copy()

    for count in range(max_iterations):
        f1, f2 = func(x, y)
        f1_xp, f2_xp = func(x + delta_x, y)
        f1_xm, f2_xm = func(x - delta_x, y)
        f1_yp, f2_yp = func(x, y + delta_y)
        f1_ym, f2_ym = func(x, y - delta_y)

        jacobian_11 = (f1_xp - f1_xm) / (2 * delta_x)
        jacobian_12 = (f1_yp - f1_ym) / (2 * delta_y)
        jacobian_21 = (f2_xp - f2_xm) / (2 * delta_x)
        jacobian_22 = (f2_yp - f2_ym) / (2 * delta_y)

        det = jacobian_11 * jacobian_22 - jacobian_12 * jacobian_21
        inv_jacobian_det = np.zeros_like(det)
        mask_det = det != 0
        inv_jacobian_det[mask_det] = 1.0 / det[mask_det]

        err_abs_x = np.zeros_like(x)
        err_abs_y = np.zeros_like(y)
        err_abs_x[mask_det] = inv_jacobian_det[mask_det] * (
            jacobian_22[mask_det] * f1[mask_det] - jacobian_12[mask_det] * f2[mask_det]
        )
        err_abs_y[mask_det] = inv_jacobian_det[mask_det] * (
            jacobian_11[mask_det] * f2[mask_det] - jacobian_21[mask_det] * f1[mask_det]
        )

        x -= err_abs_x
        y -= err_abs_y

        err_rel_x = np.abs(err_abs_x / x)
        err_rel_y = np.abs(err_abs_y / y)
        rel_err = max(np.nanmax(err_rel_x), np.nanmax(err_rel_y))
        if rel_err <= relative_tolerance:
            break

    return x, y, count + 1, rel_err


def depends_on_optional(module_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            spec = find_spec(module_name)
            if spec is None:
                raise ImportError(
                    f"Optional dependency {module_name} not found ({func.__name__})."
                )
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
