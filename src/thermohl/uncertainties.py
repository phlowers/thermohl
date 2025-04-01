# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# mypy: ignore-errors
"""Tools to perform Monte Carlo simulations using the thermOHL steady solvers with uncertain input parameters."""

from typing import Union, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import circmean
from scipy.stats import circstd
from thermohl import distributions, frozen_dist
from thermohl import solver
from thermohl import utils


def default_uncertainties() -> dict:
    """
    Get default parameters for uncertainties.

    Returns
    -------
    dict
        Dictionnary of default distributions and parameters.

    """
    return utils.add_default_uncertainties({}, warning=False)


def cummean(x: np.ndarray) -> np.ndarray:
    """Cumulative mean."""
    return np.cumsum(x) / (1 + np.array(range(len(x))))


def cumstd(x: np.ndarray) -> np.ndarray:
    """Cumulative std."""
    return np.sqrt(cummean(x**2) - cummean(x) ** 2)


def _get_dist(du: dict, mean: float) -> frozen_dist:
    """Get distribution
    -- based on parameters in dict du and with mean mean"""
    mu = mean
    # set std
    sigma = du["std"]
    if du["relative_std"]:
        sigma *= mean
    # if std is 0., return uniform
    if sigma == 0.0:
        return scipy.stats.uniform(mean, mean)

    # select ditribution
    if du["dist"] == "truncnorm":
        a, b = du["min"], du["max"]
        dist = distributions.truncnorm(a, b, mu, sigma)
    elif du["dist"] == "vonmises":
        dist = distributions.vonmises(np.deg2rad(mu), np.deg2rad(sigma))
    elif du["dist"] == "wrapnorm":
        dist = distributions.wrapnorm(np.deg2rad(mu), np.deg2rad(sigma))
    else:
        raise ValueError("Dist keyword not supported")

    return dist


def _generate_samples(
    dc: dict, i: int, du: dict, ns: int, check: bool = False
) -> Union[dict, Tuple[dict, pd.DataFrame]]:
    """
    Generate random samples for all input parameters affected with a probability distribution.
    """

    # sample dict
    ds = {}
    # check dataframe
    if check:
        cl = [
            "key",
            "dist",
            "mean",
            "std",
            "min",
            "max",
            "s_mean",
            "s_std",
            "s_min",
            "s_max",
            "circular",
        ]
        dk = pd.DataFrame(columns=cl, data=np.zeros((len(dc), len(cl))) * np.nan)
        dk.loc[:, "circular"] = False

    # loop on dict
    for j, k in enumerate(dc):
        if k not in du.keys():
            continue
        dist = du[k]["dist"]
        mean = dc[k][i]
        if dist is None or np.isnan(mean):
            sample = mean * np.ones((ns,), dtype=type(mean))
        else:
            sample = _get_dist(du[k], mean).rvs(ns)
            if dist == "vonmises" or dist == "wrapnorm":
                sample = np.rad2deg(sample) % 360.0
        ds[k] = sample

        if check:
            dk.loc[j, "key"] = k
            dk.loc[j, "dist"] = dist
            dk.loc[j, "mean"] = mean
            if "std" in du[k].keys():
                dk.loc[j, "std"] = du[k]["std"]
                if du[k]["relative_std"]:
                    dk.loc[j, "std"] *= mean
            if "min" in du[k].keys():
                dk.loc[j, "min"] = du[k]["min"]
            if "max" in du[k].keys():
                dk.loc[j, "max"] = du[k]["max"]
            if dist in ["vonmises", "wrapnorm"]:
                dk.loc[j, "s_mean"] = circmean(sample, high=360.0, low=0.0)
                dk.loc[j, "s_std"] = circstd(sample, high=360.0, low=0.0)
                dk.loc[j, "circular"] = True
            else:
                dk.loc[j, "s_mean"] = sample.mean()
                dk.loc[j, "s_std"] = sample.std()
            dk.loc[j, "s_min"] = sample.min()
            dk.loc[j, "s_max"] = sample.max()

    if check:
        return ds, dk

    return ds


def _rdict(
    mode: str, target: str, return_surf: bool, return_core: bool, return_avg: bool
) -> dict:
    """Code factorization"""
    if mode == "temperature":
        rdc = dict(return_core=return_core, return_avg=return_avg, return_power=False)
    elif mode == "intensity":
        rdc = dict(
            target=target,
            return_core=return_core,
            return_avg=return_avg,
            return_surf=return_surf,
            return_power=False,
        )
    else:
        raise ValueError("")
    return rdc


def _compute(mode: str, s: solver.Solver, tmx: Union[float, np.ndarray], rdc: dict):
    """Code factorization"""
    if mode == "temperature":
        r = s.steady_temperature(**rdc)
    elif mode == "intensity":
        r = s.steady_intensity(tmx, **rdc)
    else:
        raise ValueError()
    return r


def _steady_uncertainties(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    ns: int,
    return_surf: bool,
    return_core: bool,
    return_avg: bool,
    return_raw: bool,
    mode: str = "temperature",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """Code factorization"""
    # return dict
    rdc = _rdict(mode, target, return_surf, return_core, return_avg)

    # save solver dict
    dsave = s.dc

    #  all to max_len size
    n = utils.dict_max_len(s.dc)
    dc = utils.extend_to_max_len(s.dc, n)
    Tmax = tmax * np.ones(
        n,
    )

    # add missing uncertainties parameters
    du = utils.add_default_uncertainties(u)

    # init outputs
    rl = []
    s.dc = solver.default_values()
    tmp = _compute(mode, s, 99.0, rdc)
    cl = []
    for c in tmp.columns:
        cl.append(c + "_mean")
        cl.append(c + "_std")
    dr = pd.DataFrame(data=np.zeros((n, len(cl))), columns=cl)

    # for each entry, generate sample then compute
    for i in range(n):
        s.dc = _generate_samples(dc, i, du, ns, check=False)
        r = _compute(mode, s, Tmax[i], rdc)
        if return_raw:
            rl.append(r)
        mu = r.mean()
        sg = r.std()
        for c in r.columns:
            dr.loc[i, c + "_mean"] = mu[c]
            dr.loc[i, c + "_std"] = sg[c]

    # restore solver dict
    s.dc = dsave

    if return_raw:
        return dr, rl
    else:
        return dr


def temperature(
    s: solver.Solver,
    u: dict = {},
    ns: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_raw: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Perform Monte Carlo simulation using the steady temperature solver.
    """
    return _steady_uncertainties(
        s,
        np.nan,
        None,
        u,
        ns,
        None,
        return_core,
        return_avg,
        return_raw,
        mode="temperature",
    )


def intensity(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str = "surf",
    u: dict = {},
    ns: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_surf: bool = False,
    return_raw: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Perform Monte Carlo simulation using the steady intensity solver.
    """
    return _steady_uncertainties(
        s,
        tmax,
        target,
        u,
        ns,
        return_surf,
        return_core,
        return_avg,
        return_raw,
        mode="intensity",
    )


def _diff_method(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    q: float = 0.95,
    return_surf: bool = False,
    return_core: bool = False,
    return_avg: bool = False,
    ep: float = 1.0e-06,
    mode: str = "temperature",
) -> pd.DataFrame:
    """."""
    # return dict
    rdc = _rdict(
        mode,
        target,
        return_surf=return_surf,
        return_core=return_core,
        return_avg=return_avg,
    )

    # save solver dict
    dsave = s.dc

    #  all to max_len size
    n = utils.dict_max_len(s.dc)
    dc = utils.extend_to_max_len(s.dc, n)

    # add missing uncertainties parameters
    du = utils.add_default_uncertainties(u)

    y0 = _compute(mode, s, tmax, rdc)
    dr = y0 * 0.0
    for k in dc:
        if k not in du.keys() or du[k]["dist"] is None:
            continue
        if np.all(np.isnan(dc[k])):
            continue

        mu = dc[k]
        dx = np.zeros_like(mu)
        for i in range(n):
            if np.isnan(mu[i]):
                dx[i] = 0.0
            else:
                dist = _get_dist(du[k], mu[i])
                try:
                    dx[i] = 0.5 * np.diff(dist.ppf([0.5 * (1 - q), 0.5 * (1 + q)]))[0]
                except ValueError:
                    dx[i] = 0.0
                if np.isnan(dx[i]):
                    dx[i] = 0.0
        s.dc[k] = mu + ep
        yp = _compute(mode, s, tmax, rdc)
        s.dc[k] = mu
        dy = np.abs(yp - y0) / ep
        for c in dy.columns:
            dy.loc[:, c] *= dx
        dr += dy
        if np.any(dy.isna()):
            print("Nans with key %s" % (k,))

    dq = pd.DataFrame()
    for c in y0.columns:
        dq.loc[:, c] = y0.loc[:, c]
        dq.loc[:, c + "_delta"] = dr.loc[:, c]

    # restore solver dict
    s.dc = dsave

    return dq


def temperature_diff(
    s: solver.Solver,
    u: dict,
    q: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    ep: float = 1.0e-06,
) -> pd.DataFrame:
    """."""
    return _diff_method(
        s,
        np.nan,
        None,
        u,
        q=q,
        return_core=return_core,
        return_avg=return_avg,
        ep=ep,
        mode="temperature",
    )


def intensity_diff(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    q: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    return_surf: bool = False,
    ep: float = 1.0e-06,
):
    """."""
    return _diff_method(
        s,
        tmax,
        target,
        u,
        q=q,
        return_core=return_core,
        return_avg=return_avg,
        return_surf=return_surf,
        ep=ep,
        mode="intensity",
    )


def sensitivity(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    u: dict,
    ns: int,
    target: str,
    return_surf: bool,
    return_core: bool,
    return_avg: bool,
    mode: str = "temperature",
) -> Tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Perform a sensitivity analysis with Sobol indices (first order and total indices) using the Monte Carlo method.
    """
    # return dict
    rdc = _rdict(mode, target, return_surf, return_core, return_avg)

    # save solver dict
    dsave = s.dc

    #  all to max_len size
    n = utils.dict_max_len(s.dc)
    dc = utils.extend_to_max_len(s.dc, n)
    Tmax = tmax * np.ones(
        n,
    )

    # add missing uncertainties parameters
    du = utils.add_default_uncertainties(u)

    # init outputs
    d1l = []
    dtl = []

    # for each entry, compute
    for i in range(n):
        # first sample
        smp = _generate_samples(dc, i, du, ns, check=False)
        s.dc = smp
        dp = _compute(mode, s, Tmax[i], rdc)

        # second sample
        smq = _generate_samples(dc, i, du, ns, check=False)
        s.dc = smq
        dq = _compute(mode, s, Tmax[i], rdc)

        pqs = ((dp - dq) ** 2).sum()

        #
        d1 = pd.DataFrame(
            columns=["var"] + dp.columns.tolist(),
            data=np.zeros((len(du), 1 + len(dp.columns))),
        )
        d1.loc[:, "var"] = du.keys()
        dt = pd.DataFrame(
            columns=["var"] + dp.columns.tolist(),
            data=np.zeros((len(du), 1 + len(dp.columns))),
        )
        dt.loc[:, "var"] = du.keys()

        # mix samples, run and compute 1st and total indexes
        for j, k in enumerate(du):
            s.dc = smp.copy()
            s.dc[k] = smq[k]
            dpj = _compute(mode, s, Tmax[i], rdc)

            s.dc = smq.copy()
            s.dc[k] = smp[k]
            dqj = _compute(mode, s, Tmax[i], rdc)

            denom = pqs + ((dpj - dqj) ** 2).sum()
            d1.iloc[j, 1:] = 2.0 * ((dqj - dq) * (dp - dpj)).sum() / denom
            dt.iloc[j, 1:] = ((dq - dqj) ** 2 + (dp - dpj) ** 2).sum() / denom

        d1l.append(d1)
        dtl.append(dt)

    # restore solver dict
    s.dc = dsave

    return d1l, dtl
