"""Power terms implementation using RTE's olla project choices.

Based on NT-RD-CNER-DL-SLA-20-00215 by RTE.
"""
from typing import Union

import numpy as np

from thermohl.power import ieee
from thermohl.power.base import PowerTerm
from thermohl.power.base import RadiativeCooling as RadiativeCooling_


class JouleHeating(PowerTerm):
    """Joule heating term."""

    def __init__(
            self,
            I: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            l: Union[float, np.ndarray],
            T20: Union[float, np.ndarray] = 20.,
            f: Union[float, np.ndarray] = 50.,
            **kwargs
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        I : float or np.ndarray
            Transit intensity.
        D : float or np.ndarray
            External diameter.
        d : float or np.ndarray
            core diameter.
        A : float or np.ndarray
            External (total) section.
        a : float or np.ndarray
            core section.
        km : float or np.ndarray
            Coefficient for magnetic effects.
        ki : float or np.ndarray
            Coefficient for magnetic effects.
        kl : float or np.ndarray
            Linear resistance augmentation with temperature.
        kq : float or np.ndarray
            Quadratic resistance augmentation with temperature.
        RDC20 : float or np.ndarray
            Electric resistance per unit length (DC) at 20°C.
        l : float or np.ndarray
            ...
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.
        f : float or np.ndarray, optional
            Current frequency (Hz). The default is 50.

        """
        self.I = I
        self.D = D
        self.d = d
        # self.A = A
        # self.a = a
        # self.km = km
        self.kem = self._kem(A, a, km, ki)
        # self.ki = ki
        self.kl = kl
        self.kq = kq
        self.RDC20 = RDC20
        # self.l = l
        self.T20 = T20
        self.f = f

    def _rdc(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute resistance per unit length for direct current."""
        dt = (T - self.T20)
        return self.RDC20 * (1. + self.kl * dt + self.kq * dt**2)

    def _ks(self, rdc):
        """Compute skin-effect coefficient."""
        # Note: approx version as in [NT-RD-CNER-DL-SLA-20-00215]
        z = 8 * np.pi * self.f * (self.D - self.d)**2 / ((self.D**2 - self.d**2) * 1.0E+07 * rdc)
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1. - self.d / self.D
        return 1 + a * (1. - 0.5 * beta - b * beta**2)

    def _kem(self, A, a, km, ki) -> Union[float, np.ndarray]:
        """Compute magnetic coefficient."""
        s = np.ones_like(self.I) * np.ones_like(A) * np.ones_like(a) * np.ones_like(km) * np.ones_like(ki)
        z = s.shape == ()
        if z:
            s = np.array([1.])
        I_ = self.I * s
        a_ = a * s
        A_ = A * s
        m = a_ > 0.
        ki_ = ki * s
        kem = km * s
        kem[m] += ki_[m] * I_[m] / ((A_[m] - a_[m]) * 1.0E+06)
        if z:
            kem = kem[0]
        return kem

    def value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        rdc = self._rdc(T)
        ks = self._ks(rdc)
        rac = self.kem * ks * rdc
        return rac * self.I**2


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def __init__(
            self,
            lat: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray],
            srad: Union[float, np.ndarray] = None,
            **kwargs,
    ):
        r"""Init with args.

        See ieee.SolarHeating; it is exactly the same with altitude and
        turbidity set to zero. If more than one input are numpy arrays, they
        should have the same size.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude.
        azm : float or np.ndarray
            Azimuth.
        month : int or np.ndarray
            Month number (must be between 1 and 12).
        day : int or np.ndarray
            Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
            month).
        hour : float or np.ndarray
            Hour of the day (solar, must be between 0 and 23).
        D : float or np.ndarray
            external diameter.
        alpha : float or np.ndarray
            Solar absorption coefficient.
        srad : xxx
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        alt = 0.
        tb = 0.
        super().__init__(lat, alt, azm, tb, month, day, hour, D, alpha, srad)


class ConvectiveCooling(ieee.ConvectiveCooling):
    pass


class RadiativeCooling(RadiativeCooling_):
    pass
