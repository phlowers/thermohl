"""Base class to build a solver for heat equation."""

from typing import Union

import numpy as np

from thermohl.power.base import PowerTerm


class _DEFPARAM:
    tmin = -99.
    tmax = +999.
    tol = 1.0E-06
    maxiter = 64
    imin = 0.
    imax = 9999.


class Args:
    """Object to store Solver args in a dict-like manner."""

    # __slots__ = [
    #     'lat', 'lon', 'alt', 'azm', 'month', 'day', 'hour', 'Ta', 'Pa', 'rh', 'pr', 'ws', 'wa', 'al', 'tb', 'I', 'm',
    #     'd', 'D', 'a', 'A', 'R', 'l', 'c', 'alpha', 'epsilon', 'RDC20', 'km', 'ki', 'kl', 'kq', 'RDCHigh', 'RDCLow',
    #     'THigh', 'TLow'
    # ]

    def __init__(self, dic: dict = {}):
        # add default values
        self._set_default_values()
        # use values from input dict
        keys = self.keys()
        for k in dic:
            if k in keys and dic[k] is not None:
                self[k] = dic[k]

    def _set_default_values(self):
        """Set default values."""

        self.lat = 45.  # latitude (deg)
        self.lon = 0.  # longitude (deg)
        self.alt = 0.  # altitude (m)
        self.azm = 0.  # azimuth (deg)

        self.month = 3  # month number (1=Jan, 2=Feb, ...)
        self.day = 21  # day of the month
        self.hour = 12  # hour of the day (in [0, 23] range)

        self.Ta = 15.  # ambient temperature (C)
        self.Pa = 1.0E+05  # ambient pressure (Pa)
        self.rh = 0.8  # relative humidity (none, in [0, 1])
        self.pr = 0.  # rain precipitation rate (m.s**-1)
        self.ws = 0.  # wind speed (m.s**-1)
        self.wa = 90.  # wind angle (deg, regarding north)
        self.al = 0.8  # albedo (1)
        self.tb = 0.1  # coefficient for air pollution from 0 (clean) to 1 (polluted)

        self.I = 100.  # transit intensity (A)

        self.m = 1.5  # mass per unit length (kg.m**-1)
        self.d = 1.9E-02  # core diameter (m)
        self.D = 3.0E-02  # external (global) diameter (m)
        self.a = 2.84E-04  # core section (m**2)
        self.A = 7.07E-04  # external (global) section (m**2)
        self.R = 4.0E-02  # roughness (1)
        self.l = 1.0  # radial thermal conductivity (W.m**-1.K**-1)
        self.c = 500.  # specific heat capacity (J.kg**-1.K**-1)

        self.alpha = 0.5  # solar absorption (1)
        self.epsilon = 0.5  # emissivity (1)
        self.RDC20 = 2.5E-05  # electric resistance per unit length (DC) at 20°C (Ohm.m**-1)
        self.km = 1.006  # coefficient for magnetic effects (1)
        self.ki = 0.016  # coefficient for magnetic effects (A**-1)
        self.kl = 3.8E-03  # linear resistance augmentation with temperature (K**-1)
        self.kq = 8.0E-07  # quadratic resistance augmentation with temperature (K**-2)
        self.RDCHigh = 3.05E-05  # electric resistance per unit length (DC) at THigh (Ohm.m**-1)
        self.RDCLow = 2.66E-05  # electric resistance per unit length (DC) at TLow (Ohm.m**-1)
        self.THigh = 60.  # temperature for RDCHigh measurement (°C)
        self.TLow = 20.  # temperature for RDCLow measurement (°C)

    def keys(self):
        """Get list of members as dict keys."""
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def max_len(self):
        """."""
        n = 1
        for k in self.keys():
            try:
                n = max(n, len(self[k]))
            except TypeError:
                pass
        return n

    def extend_to_max_len(self, inplace: bool = False):
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


class Solver:
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and three cooling terms (convective,
    radiative and precipitation cooling). This class is used to solve a
    temperature problem with the heating and cooling terms passed to its
    __init__ function.
    """

    def __init__(self, dic: dict = None, joule: PowerTerm = PowerTerm(), solar: PowerTerm = PowerTerm(),
                 convective: PowerTerm = PowerTerm(), radiative: PowerTerm = PowerTerm(),
                 precipitation: PowerTerm = PowerTerm()):
        if dic is None:
            dic = {}
        self.args = Args(dic)
        self.jh = joule
        self.sh = solar
        self.cc = convective
        self.rc = radiative
        self.pc = precipitation
        return

    def balance(self, T=Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (
                self.jh.value(T, self.args) +
                self.sh.value(T, self.args) -
                self.cc.value(T, self.args) -
                self.rc.value(T, self.args) -
                self.pc.value(T, self.args)
        )

    def steady_temperature(self):
        return

    def transient_temperature(self):
        return

    def steady_intensity(self):
        return
