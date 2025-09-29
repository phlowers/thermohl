# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Base class to build a solver for heat equation."""

import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Type, Any, Optional, KeysView, Dict

import numpy as np
import pandas as pd

from thermohl import (
    floatArrayLike,
    floatArray,
    intArray,
    numberArray,
    numberArrayLike,
)
from thermohl.power import PowerTerm


class _DEFPARAM:
    tmin = -99.0
    tmax = +999.0
    tol = 1.0e-09
    maxiter = 64
    imin = 0.0
    imax = 9999.0


class Args:
    """Object to store Solver args in a dict-like manner."""

    # __slots__ = [
    #     'lat', 'lon', 'alt', 'azm', 'month', 'day', 'hour', 'Ta', 'Pa', 'rh', 'pr', 'ws', 'wa', 'al', 'tb', 'I', 'm',
    #     'd', 'D', 'a', 'A', 'R', 'l', 'c', 'alpha', 'epsilon', 'RDC20', 'km', 'ki', 'kl', 'kq', 'RDCHigh', 'RDCLow',
    #     'THigh', 'TLow'
    # ]

    def __init__(self, dic: Optional[dict[str, Any]] = None):
        # add default values
        self._set_default_values()
        # use values from input dict
        if dic is None:
            dic = {}
        keys = self.keys()
        for k in dic:
            if k in keys and dic[k] is not None:
                self[k] = dic[k]
        # check for shape incompatibilities
        _ = self.shape()

    def _set_default_values(self) -> None:
        """Set default values."""

        self.lat = 45.0  # latitude (deg)
        self.lon = 0.0  # longitude (deg)
        self.alt = 0.0  # altitude (m)
        self.azm = 0.0  # azimuth (deg)

        self.month = 3  # month number (1=Jan, 2=Feb ...)
        self.day = 21  # day of the month
        self.hour = 12.0  # hour of the day (in [0, 23] range)

        self.Ta = 15.0  # ambient temperature (C)
        self.Pa = 1.0e05  # ambient pressure (Pa)
        self.rh = 0.8  # relative humidity (none, in [0, 1])
        self.pr = 0.0  # rain precipitation rate (m.s**-1)
        self.ws = 0.0  # wind speed (m.s**-1)
        self.wa = 90.0  # wind angle (deg, regarding north)
        self.al = 0.8  # albedo (1)
        self.tb = 0.1  # turbidity (coefficient for air pollution from 0 -clean- to 1 -polluted-)
        self.srad = float("nan")  # solar irradiance (in W.m**-2)

        self.I = 100.0  # transit intensity (A)

        self.m = 1.5  # mass per unit length (kg.m**-1)
        self.d = 1.9e-02  # core diameter (m)
        self.D = 3.0e-02  # external (global) diameter (m)
        self.a = 2.84e-04  # core section (m**2)
        self.A = 7.07e-04  # external (global) section (m**2)
        self.R = 4.0e-02  # roughness (1)
        self.l = 1.0  # radial thermal conductivity (W.m**-1.K**-1)
        self.c = 500.0  # specific heat capacity (J.kg**-1.K**-1)

        self.alpha = 0.5  # solar absorption (1)
        self.epsilon = 0.5  # emissivity (1)
        # electric resistance per unit length (DC) at 20°C (Ohm.m**-1)
        self.RDC20 = 2.5e-05

        self.km = 1.006  # coefficient for magnetic effects (1)
        self.ki = 0.016  # coefficient for magnetic effects (A**-1)
        # linear resistance augmentation with temperature (K**-1)
        self.kl = 3.8e-03
        # quadratic resistance augmentation with temperature (K**-2)
        self.kq = 8.0e-07
        # electric resistance per unit length (DC) at THigh (Ohm.m**-1)
        self.RDCHigh = 3.05e-05
        # electric resistance per unit length (DC) at TLow (Ohm.m**-1)
        self.RDCLow = 2.66e-05
        self.THigh = 60.0  # temperature for RDCHigh measurement (°C)
        self.TLow = 20.0  # temperature for RDCLow measurement (°C)

    def keys(self) -> KeysView[str]:
        """Get list of members as dict keys."""
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def shape(self) -> Tuple[int, ...]:
        """
        Compute the maximum effective shape of values in current instance.

        Members of Args can be float of arrays. If arrays, they must be
        one-dimensional. Float and 1d array can coexist, but all arrays should
        have the same shape/size.

        This method iterates over all keys in the instance's __dict__ and
        computes the maximum length of the values associated with those keys.

        If incompatible shapes are encountered, an exception is raised (ValueError).

        """
        shape_ = ()
        for k in self.keys():
            s = np.array(self[k]).shape
            d = len(s)
            er = f"Key {k} has a {s} shape when main shape is {shape_}"
            if d == 0:
                continue
            if d == 1:
                if shape_ == ():
                    shape_ = s
                elif len(shape_) == 1:
                    if shape_ != s:
                        raise ValueError(er)
                else:
                    raise ValueError(er)
            else:
                raise ValueError(
                    f"Key {k} has a {s} shape, only float and 1-dim arrays are accepted"
                )
        return shape_

    def extend(self, shape: Tuple[int, ...] = None) -> None:
        # get shape
        if shape is None:
            shape = self.shape()
        # complete if necessary
        if len(shape) == 0:
            shape = (1,)
        # create a copy dict with scaled array
        for k in self.keys():
            a = np.array(self[k])
            self[k] = a * np.ones(shape, dtype=a.dtype)

    def compress(self) -> None:
        """
        Compresses the values in the dictionary by replacing numpy arrays with a
        single unique value if all elements in the array are the same.

        Returns:
            None
        """
        for k in self.keys():
            u = np.unique(self[k])
            if len(u) == 1:
                self[k] = u[0]


class Solver(ABC):
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and three cooling terms (convective,
    radiative and precipitation cooling). This class is used to solve a
    temperature problem with the heating and cooling terms passed to its
    __init__ function.
    """

    class Names:
        pjle = "P_joule"
        psol = "P_solar"
        pcnv = "P_convection"
        prad = "P_radiation"
        ppre = "P_precipitation"
        err = "err"
        surf = "surf"
        avg = "avg"
        core = "core"
        time = "time"
        transit = "I"
        temp = "t"
        tsurf = "t_surf"
        tavg = "t_avg"
        tcore = "t_core"

        @staticmethod
        def powers() -> tuple[str, str, str, str, str]:
            return (
                Solver.Names.pjle,
                Solver.Names.psol,
                Solver.Names.pcnv,
                Solver.Names.prad,
                Solver.Names.ppre,
            )

    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ) -> None:
        """Create a Solver object.

        Args:
            dic (dict[str, Any] | None): Input values used in power terms. If there is a missing value, a default is used.
            joule (Type[PowerTerm]): Joule heating term class.
            solar (Type[PowerTerm]): Solar heating term class.
            convective (Type[PowerTerm]): Convective cooling term class.
            radiative (Type[PowerTerm]): Radiative cooling term class.
            precipitation (Type[PowerTerm]): Precipitation cooling term class.

        Returns:
            None

        """
        self.args = Args(dic)
        self.args.extend()
        self.jh = joule(**self.args.__dict__)
        self.sh = solar(**self.args.__dict__)
        self.cc = convective(**self.args.__dict__)
        self.rc = radiative(**self.args.__dict__)
        self.pc = precipitation(**self.args.__dict__)
        self.args.compress()

    def _min_shape(self) -> Tuple[int, ...]:
        shape = self.args.shape()
        if shape == ():
            shape = (1,)
        return shape

    def _transient_process_dynamic(
        self, time: np.ndarray, n: int, dynamic: dict = None
    ):
        """Code factorization for transient temperature computations.

        This methods prepare a dict with dynamic values to use in the
        compute time loop.
        """
        if len(time) < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get month, day and hours in range with time
        month, day, hour = _set_dates(
            self.args.month, self.args.day, self.args.hour, time, n
        )

        # put dynamic values in a separate dict which will be used
        # through the time loop
        dynamic_ = {
            "month": month,
            "day": day,
            "hour": hour,
        }

        if dynamic is None:
            dynamic = {}
        for k, v in dynamic.items():
            dynamic_[k] = reshape(v, len(time), n)

        return dynamic_

    def update(self) -> None:
        self.args.extend()
        self.jh.__init__(**self.args.__dict__)
        self.sh.__init__(**self.args.__dict__)
        self.cc.__init__(**self.args.__dict__)
        self.rc.__init__(**self.args.__dict__)
        self.pc.__init__(**self.args.__dict__)
        self.args.compress()

    def balance(self, T: floatArrayLike) -> floatArrayLike:
        return (
            self.jh.value(T)
            + self.sh.value(T)
            - self.cc.value(T)
            - self.rc.value(T)
            - self.pc.value(T)
        )

    @abstractmethod
    def steady_temperature(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def transient_temperature(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def steady_intensity(self) -> pd.DataFrame:
        raise NotImplementedError


def reshape(input_var: numberArrayLike, nb_row: int, nb_columns: int) -> numberArray:
    """
    Reshape the input array to the specified dimensions (nr, nc) if possible.

    Args:
        input_var (numberArrayLike): Variable to be reshaped.
        nb_row (int): Desired number of rows for the reshaped array.
        nb_columns (int): Desired number of columns for the reshaped array.

    Returns:
        numberArray: Reshaped array of size (nb_row, nb_columns). If reshaping is not possible,
            returns an array filled with the input_value repeated to fill the dimension (nb_row, nb_columns).

    Raises:
        ValueError: If the input has an invalid shape that cannot be reshaped.
    """

    input_array = np.array(input_var)
    input_shape = input_array.shape

    msg = f"Input array has incompatible shape {input_shape} with specified number of rows ({nb_row}) and/or columns ({nb_columns})."

    if len(input_shape) == 0:
        reshaped_array = input_array * np.ones(
            (nb_row, nb_columns), dtype=input_array.dtype
        )
    elif len(input_shape) == 1:
        if nb_row == input_shape[0]:
            reshaped_array = np.column_stack(nb_columns * (input_array,))
        elif nb_columns == input_shape[0]:
            reshaped_array = np.vstack(nb_row * (input_array,))
        else:
            raise ValueError(msg)
    elif input_shape == (nb_row, nb_columns):
        return input_array
    else:
        raise ValueError(msg)
    return reshaped_array


def _set_dates(
    month: floatArrayLike,
    day: floatArrayLike,
    hour: floatArrayLike,
    time: floatArray,
    n: int,
) -> Tuple[intArray, intArray, floatArray]:
    """
    Set months, days and hours as 2D arrays.

    This function is used in transient temperature computations. Inputs month,
    day and hour are floats or 1D arrays of size n; input t is a time vector of
    size N with evaluation times in seconds. It sets arrays months, days and
    hours, of size (N, n) such that
        months[i, j] = datetime(month[j], day[j], hour[j]) + t[i] .

    Args:
        month (floatArrayLike): Array of floats or float representing the months.
        day (floatArrayLike): Array of floats or float representing the days.
        hour (floatArrayLike): Array of floats or float representing the hours.
        time (floatArray): Array of floats representing the time vector in seconds.
        n (int): Size of the input arrays month, day, and hour.

    Returns:
    Tuple[intArray, intArray, floatArray]:
        - months (intArray): 2D array of shape (N, n) with month values.
        - days (intArray): 2D array of shape (N, n) with day values.
        - hours (floatArray): 2D array of shape (N, n) with hour values.
    """
    oi = np.ones((n,), dtype=int)
    of = np.ones((n,), dtype=float)
    month2 = month * oi
    day2 = day * oi
    hour2 = hour * of

    N = len(time)
    months = np.zeros((N, n), dtype=int)
    days = np.zeros((N, n), dtype=int)
    hours = np.zeros((N, n), dtype=float)

    td = np.array(
        [datetime.timedelta()]
        + [datetime.timedelta(seconds=float(time[i] - time[0])) for i in range(1, N)]
    )

    for j in range(n):
        hj = int(np.floor(hour2[j]))
        dj = datetime.timedelta(seconds=float(3600.0 * (hour2[j] - hj)))
        t0 = datetime.datetime(year=2000, month=month2[j], day=day2[j], hour=hj) + dj
        ts = pd.Series(t0 + td)
        months[:, j] = ts.dt.month
        days[:, j] = ts.dt.day
        hours[:, j] = (
            ts.dt.hour
            + ts.dt.minute / 60.0
            + (ts.dt.second + ts.dt.microsecond * 1.0e-06) / 3600.0
        )

    return months, days, hours
