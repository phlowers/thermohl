"""Models to compute equilibrium temperature or max intensity in a conductor."""
import datetime
from typing import Union, Tuple

import numpy as np
import pandas as pd
import thermohl.cigre as cig_
import thermohl.cner as cnr_
import thermohl.ieee as i3e_
import thermohl.olla as ola_

import thermohl.utils as utils
from thermohl.solver.base import Args, Solver


def default_values():
    """
    Get default values used in Solver class.

    Returns
    -------
    dict
        Dictionary of default values.

    """
    return utils.add_default_parameters({}, warning=False)


def _set_dates(month: Union[float, np.ndarray], day: Union[float, np.ndarray], hour: Union[float, np.ndarray],
               t: Union[float, np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set months, days and hours as 2D arrays.

    This function is used in transient temperature computations. Inputs month,
    day and hour are floats or 1D arrays of size n; input t is a time vector of
    size N with evaluation times in seconds. It sets arrays months, days and
    hours, of size (N, n) such that
        months[i, j] = datetime(month[j], day[j], hour[j]) + t[i] .
    """
    month2 = month * np.ones((n,), dtype=int)
    day2 = day * np.ones((n,), dtype=int)
    hour2 = hour * np.ones((n,), dtype=float)

    N = len(t)
    months = np.zeros((N, n), dtype=int)
    days = np.zeros((N, n), dtype=int)
    hours = np.zeros((N, n), dtype=float)

    td = np.array([datetime.timedelta()] + [datetime.timedelta(seconds=t[i] - t[i - 1]) for i in range(1, N)])

    for j in range(n):
        hj = int(np.floor(hour2[j]))
        dj = datetime.timedelta(seconds=3600. * (hour2[j] - hj))
        t0 = datetime.datetime(year=2000, month=month2[j], day=day2[j], hour=hj) + dj
        ts = pd.Series(t0 + td)
        months[:, j] = ts.dt.month
        days[:, j] = ts.dt.day
        hours[:, j] = ts.dt.hour + ts.dt.minute / 60. + (ts.dt.second + ts.dt.microsecond * 1.0E-06) / 3600.

    return months, days, hours


def cigre(dct: dict = {}):
    """
    Get a Solver using CIGRE models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, cig_.JouleHeating, cig_.SolarHeating,
                  cig_.ConvectiveCooling, cig_.RadiativeCooling)


def ieee(dct: dict = {}):
    """
    Get a Solver using IEEE models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, i3e_.JouleHeating, i3e_.SolarHeating,
                  i3e_.ConvectiveCooling, i3e_.RadiativeCooling)


def olla(dct: dict = {}, multi: bool = False):
    """
    Get a Solver using RTE-olla models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.
    multi: bool, optional
        Use multi-temp model within bisection iterations. The default is False.

    """
    if multi:
        return Solver(dct, ola_.JouleHeatingMulti, ola_.SolarHeating,
                      ola_.ConvectiveCooling, ola_.RadiativeCooling)
    else:
        return Solver(dct, ola_.JouleHeating, ola_.SolarHeating,
                      ola_.ConvectiveCooling, ola_.RadiativeCooling)


def cner(dct: dict = {}):
    """
    Get a Solver using RTE-cner models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, cnr_.JouleHeating, cnr_.SolarHeating,
                  cnr_.ConvectiveCooling, cnr_.RadiativeCooling)
