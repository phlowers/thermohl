"""Base class to build a solver for heat equation."""

import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Type, Any

import numpy as np
import pandas as pd

from thermohl import floatArrayLike
from thermohl.power.base import PowerTerm


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

    def __init__(self, dic: dict = None):
        # add default values
        self._set_default_values()
        # use values from input dict
        if dic is None:
            dic = {}
        keys = self.keys()
        for k in dic:
            if k in keys and dic[k] is not None:
                self[k] = dic[k]

    def _set_default_values(self):
        """Set default values."""

        self.lat = 45.0  # latitude (deg)
        self.lon = 0.0  # longitude (deg)
        self.alt = 0.0  # altitude (m)
        self.azm = 0.0  # azimuth (deg)

        self.month = 3  # month number (1=Jan, 2=Feb, ...)
        self.day = 21  # day of the month
        self.hour = 12  # hour of the day (in [0, 23] range)

        self.Ta = 15.0  # ambient temperature (C)
        self.Pa = 1.0e05  # ambient pressure (Pa)
        self.rh = 0.8  # relative humidity (none, in [0, 1])
        self.pr = 0.0  # rain precipitation rate (m.s**-1)
        self.ws = 0.0  # wind speed (m.s**-1)
        self.wa = 90.0  # wind angle (deg, regarding north)
        self.al = 0.8  # albedo (1)
        self.tb = 0.1  # coefficient for air pollution from 0 (clean) to 1 (polluted)

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
        self.RDC20 = (
            2.5e-05  # electric resistance per unit length (DC) at 20°C (Ohm.m**-1)
        )
        self.km = 1.006  # coefficient for magnetic effects (1)
        self.ki = 0.016  # coefficient for magnetic effects (A**-1)
        self.kl = 3.8e-03  # linear resistance augmentation with temperature (K**-1)
        self.kq = 8.0e-07  # quadratic resistance augmentation with temperature (K**-2)
        self.RDCHigh = (
            3.05e-05  # electric resistance per unit length (DC) at THigh (Ohm.m**-1)
        )
        self.RDCLow = (
            2.66e-05  # electric resistance per unit length (DC) at TLow (Ohm.m**-1)
        )
        self.THigh = 60.0  # temperature for RDCHigh measurement (°C)
        self.TLow = 20.0  # temperature for RDCLow measurement (°C)

    def keys(self):
        """Get list of members as dict keys."""
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def max_len(self) -> int:
        """."""
        n = 1
        for k in self.keys():
            try:
                n = max(n, len(self[k]))
            except TypeError:
                pass
        return n

    def extend_to_max_len(self, inplace: bool = True):
        """."""
        if inplace:
            a = self
        else:
            a = Args()
        n = self.max_len()
        for k in self.keys():
            if isinstance(self[k], np.ndarray):
                t = self[k].dtype
                c = len(self[k]) == n
            else:
                t = type(self[k])
                c = False
            if c:
                a[k] = self[k][:]
            else:
                a[k] = self[k] * np.ones((n,), dtype=t)

        if not inplace:
            return a

    def compress(self, inplace: bool = True):
        if inplace:
            a = self
        else:
            a = Args()
        for k in self.keys():
            if isinstance(self[k], np.ndarray):
                u = np.unique(self[k])
                if len(u) == 1:
                    a[k] = u[0]
                else:
                    a[k] = self[k]
            else:
                a[k] = self[k]

        if not inplace:
            return a


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
        def powers():
            return (
                Solver.Names.pjle,
                Solver.Names.psol,
                Solver.Names.pcnv,
                Solver.Names.prad,
                Solver.Names.ppre,
            )

    def __init__(
        self,
        dic: dict = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ):
        """Create a Solver object.

        Parameters
        ----------
        dc : dict
            Input values used in power terms. If there is a missing value, a
            default is used.
        jouleH : utils.PowerTerm
            Joule heating term.
        solarH : utils.PowerTerm
            Solar heating term.
        convectiveC : utils.PowerTerm
            Convective cooling term.
        radiativeC : utils.PowerTerm
            Radiative cooling term.

        Returns
        -------
        None.

        """
        if dic is None:
            dic = {}
        self.args = Args(dic)

        self.args.extend_to_max_len(inplace=True)
        self.jh = joule(**self.args.__dict__)
        self.sh = solar(**self.args.__dict__)
        self.cc = convective(**self.args.__dict__)
        self.rc = radiative(**self.args.__dict__)
        self.pc = precipitation(**self.args.__dict__)
        self.args.compress()
        return

    def update(self):
        self.args.extend_to_max_len(inplace=True)
        self.jh.__init__(**self.args.__dict__)
        self.sh.__init__(**self.args.__dict__)
        self.cc.__init__(**self.args.__dict__)
        self.rc.__init__(**self.args.__dict__)
        self.pc.__init__(**self.args.__dict__)
        self.args.compress()
        return

    def balance(self, T: floatArrayLike) -> floatArrayLike:
        return (
            self.jh.value(T)
            + self.sh.value(T)
            - self.cc.value(T)
            - self.rc.value(T)
            - self.pc.value(T)
        )

    @abstractmethod
    def steady_temperature(self):
        pass

    @abstractmethod
    def transient_temperature(self):
        pass

    @abstractmethod
    def steady_intensity(self):
        pass


def _reshape1d(v: Any, n: int) -> np.ndarray[Any]:
    """Reshape input v in size (n,) if possible."""
    try:
        l = len(v)
        if l == 1:
            w = v * np.ones(n, dtype=v.dtype)
        else:
            raise ValueError("Uncompatible size")
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
            elif len(s) == 0:
                raise AttributeError()
            else:
                w = np.reshape(v, (nr, nc))
        except AttributeError:
            w = v * np.ones((nr, nc), dtype=type(v))
    return w


def _set_dates(
    month: floatArrayLike,
    day: floatArrayLike,
    hour: floatArrayLike,
    t: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set months, days and hours as 2D arrays.

    This function is used in transient temperature computations. Inputs month,
    day and hour are floats or 1D arrays of size n; input t is a time vector of
    size N with evaluation times in seconds. It sets arrays months, days and
    hours, of size (N, n) such that
        months[i, j] = datetime(month[j], day[j], hour[j]) + t[i] .
    """
    oi = np.ones((n,), dtype=int)
    of = np.ones((n,), dtype=float)
    month2 = month * oi
    day2 = day * oi
    hour2 = hour * of

    N = len(t)
    months = np.zeros((N, n), dtype=int)
    days = np.zeros((N, n), dtype=int)
    hours = np.zeros((N, n), dtype=float)

    td = np.array(
        [datetime.timedelta()]
        + [datetime.timedelta(seconds=t[i] - t[i - 1]) for i in range(1, N)]
    )

    for j in range(n):
        hj = int(np.floor(hour2[j]))
        dj = datetime.timedelta(seconds=3600.0 * (hour2[j] - hj))
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
