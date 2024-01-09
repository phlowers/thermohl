"""Misc. numeric functions."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy._lib._util import _asarray_validated
from scipy._lib._util import _lazywhere


def _reshape1d(v, n):
    """Reshape input v in size (n,) if possible."""
    try:
        l = len(v)
        if l == 1:
            w = v * np.ones(n, dtype=v.dtype)
        else:
            raise ValueError('Uncompatible size')
    except AttributeError:
        w = v * np.ones(n, dtype=type(v))
    return w


def reshape(v, nr=None, nc=None):
    """Reshape input v in size (nr, nc) if possible."""
    if nr is None and nc is None:
        raise ValueError()
    if nr is None:
        w = _reshape1d(v, nc)
    elif nc is None:
        w = _reshape1d(v, nr)
    else:
        try:
            s = v.shape
            if len(s) == 1:
                if nr == s[0]:
                    w = np.column_stack(nc * (v,))
                elif nc == s[0]:
                    w = np.row_stack(nr * (v,))
            else:
                w = np.reshape(v, (nr, nc))
        except AttributeError:
            w = v * np.ones((nr, nc), dtype=type(v))
    return w


def vect_bisection(fun: callable, a: np.ndarray, b: np.ndarray, tol: float, maxiter: int, print_err: bool = False) \
        -> tuple[np.ndarray, np.ndarray]:
    """Hand-made bisection method (vector mode)."""
    e = np.abs(b - a)
    c = 1
    while np.nanmax(e) > tol and c <= maxiter:
        x = 0.5 * (a + b)
        y = fun(x)
        i = y > 0
        a[i] = x[i]
        b[~i] = x[~i]
        e = np.abs(b - a)
        c = c + 1
    x = 0.5 * (a + b)
    x[np.isnan(fun(x))] = np.nan
    if print_err:
        print('Bisection max err (abs) : %.2E; count=%d' % (np.max(e), c))
    return x, e


def _del2(p0, p1, d):
    """Copied from scipy.optimize."""
    return p0 - np.square(p1 - p0) / d


def _relerr(actual, desired):
    """Copied from scipy.optimize."""
    return (actual - desired) / desired


def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel, raise_err):
    """Almost copied from scipy.optimize.

    Changed
        if np.all(np.abs(relerr) < xtol)
    into
        if np.nanmax(np.abs(relerr)) < xtol

    Added an option to choose wether raising an error or not when convergence fails.

    """
    p0 = x0
    for i in range(maxiter):
        p1 = func(p0, *args)
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
        if np.nanmax(np.abs(relerr)) < xtol:
            return p
        p0 = p
    msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
    if raise_err:
        raise RuntimeError(msg)
    else:
        print(msg)
        return np.where(np.abs(relerr) < xtol, p, np.nan)


def fixed_point(func, x0, args=(), xtol=1e-8, maxiter=500, method='del2', raise_err=True):
    """Copied from scipy.optimize."""
    use_accel = {'del2': True, 'iteration': False}[method]
    x0 = _asarray_validated(x0, as_inexact=True)
    return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel, raise_err)
