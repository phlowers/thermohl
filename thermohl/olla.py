"""Power terms implementation using RTE's olla project choices.

Based on NT-RD-CNER-DL-SLA-20-00215 by RTE.
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, Tuple

from thermohl import ieee
from thermohl.numeric import fixed_point
from thermohl.utils import PowerTerm
from thermohl.utils import RadiativeCooling


class JouleHeating(PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _RDC(T: Union[float, np.ndarray], kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
             RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.) -> Union[float, np.ndarray]:
        """Compute resistance per unit length for direct current."""
        return RDC20 * (1. + kl * (T - T20) + kq * (T - T20)**2)

    @staticmethod
    def _ks(R: Union[float, np.ndarray], D: Union[float, np.ndarray],
            d: Union[float, np.ndarray], f: Union[float, np.ndarray] = 50.) -> Union[float, np.ndarray]:
        """Compute skin-effect coefficient."""
        # Note: approx version as in [NT-RD-CNER-DL-SLA-20-00215]
        z = 8 * np.pi * f * (D - d)**2 / ((D**2 - d**2) * 1.0E+07 * R)
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1. - d / D
        return 1 + a * (1. - 0.5 * beta - b * beta**2)

    @staticmethod
    def _RAC(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
             D: Union[float, np.ndarray], d: Union[float, np.ndarray],
             A: Union[float, np.ndarray], a: Union[float, np.ndarray],
             km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
             kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
             RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
             f: Union[float, np.ndarray] = 50.) -> Union[float, np.ndarray]:
        """Compute resistance per unit length for alternative current."""
        R = JouleHeating._RDC(T, kl, kq, RDC20, T20)
        I = I * np.ones_like(T)
        d = d * np.ones_like(T)
        a = a * np.ones_like(T)
        A = A * np.ones_like(T)
        m = d > 0.
        kem = km * np.ones_like(T)
        ki = ki * np.ones_like(T)
        kem[m] += ki[m] * I[m] / ((A[m] - a[m]) * 1.0E+06)
        return kem * JouleHeating._ks(R, D, d, f) * R

    @staticmethod
    def _temp_header(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                     D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                     A: Union[float, np.ndarray], a: Union[float, np.ndarray], l: Union[float, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray],
                     Union[float, np.ndarray], Union[float, np.ndarray]]:
        d = d * np.ones_like(T)
        D = D * np.ones_like(T)
        a = a * np.ones_like(T)
        A = A * np.ones_like(T)
        m = d > 0.
        c = 0.5 * np.ones_like(T)
        c[m] -= d[m]**2 / (D[m]**2 - d[m]**2) * np.log(D[m] / d[m])
        c *= 0.5 * I**2 / (np.pi * l)
        return c, d, D, a, A

    @staticmethod
    def _avg_core(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                  D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                  A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                  km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                  kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                  RDC20: Union[float, np.ndarray], l: Union[float, np.ndarray],
                  T20: Union[float, np.ndarray] = 20., f: Union[float, np.ndarray] = 50.,
                  xtol: float = 1.0E-03, maxiter: int = 128, **kwargs) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute average temperature and core temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a, l)

        def Tmoy(x) -> Union[float, np.ndarray]:
            return T + 0.5 * c * JouleHeating._RAC(x, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)

        Ta = fixed_point(Tmoy, np.zeros_like(T), xtol=xtol, maxiter=maxiter, raise_err=False)
        Tc = 2. * Ta - T

        return Ta, Tc

    @staticmethod
    def _surf_core(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                   D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                   A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                   km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                   kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                   RDC20: Union[float, np.ndarray], l: Union[float, np.ndarray],
                   T20: Union[float, np.ndarray] = 20., f: Union[float, np.ndarray] = 50., **kwargs) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute surface temperature and core temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a, l)
        Ts = T - 0.5 * c * JouleHeating._RAC(T, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
        Tc = 2. * T - Ts
        return Ts, Tc

    @staticmethod
    def _avg_surf(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                  D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                  A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                  km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                  kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                  RDC20: Union[float, np.ndarray], l: Union[float, np.ndarray],
                  T20: Union[float, np.ndarray] = 20., f: Union[float, np.ndarray] = 50.,
                  xtol: float = 1.0E-03, maxiter: int = 128, **kwargs) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute average temperature and surface temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a, l)

        def Tmoy(x) -> Union[float, np.ndarray]:
            return T - 0.5 * c * JouleHeating._RAC(x, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)

        Ta = fixed_point(Tmoy, np.zeros_like(T), xtol=xtol, maxiter=maxiter, raise_err=False)
        Ts = 2. * Ta - T

        return Ta, Ts

    def value(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
              D: Union[float, np.ndarray], d: Union[float, np.ndarray],
              A: Union[float, np.ndarray], a: Union[float, np.ndarray],
              km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
              kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
              RDC20: Union[float, np.ndarray], l: Union[float, np.ndarray],
              T20: Union[float, np.ndarray] = 20., f: Union[float, np.ndarray] = 50., **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return JouleHeating._RAC(T, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f) * I**2


class JouleHeatingMulti(JouleHeating):
    """Joule heating term (multi).

    Similar to JouleHeating, but the average and core temperature are taken
    into account in the value method, which (slightly/hardly) changes the
    results.
    """

    def value(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
              D: Union[float, np.ndarray], d: Union[float, np.ndarray],
              A: Union[float, np.ndarray], a: Union[float, np.ndarray],
              km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
              kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
              RDC20: Union[float, np.ndarray], l: Union[float, np.ndarray],
              T20: Union[float, np.ndarray] = 20., f: Union[float, np.ndarray] = 50., **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Ta, _ = JouleHeatingMulti._avg_core(T, I, D, d, A, a, km, ki, kl, kq, RDC20, l, T20, f)
        return JouleHeatingMulti._RAC(Ta, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f) * I**2


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def value(self, T: Union[float, np.ndarray],
              lat: Union[float, np.ndarray], azm: Union[float, np.ndarray],
              month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]], hour: Union[float, np.ndarray],
              D: Union[float, np.ndarray], alpha: Union[float, np.ndarray], **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Compute solar heating.

        See ieee.SolarHeating; it is exactly the same with altitude and
        turbidity set to zero. If more than one input are numpy arrays, they
        should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        _ = kwargs.pop('alt', None)
        _ = kwargs.pop('tb', None)
        return super().value(T=T, lat=lat, alt=0., azm=azm, tb=0., month=month, day=day, hour=hour, D=D, alpha=alpha,
                             **kwargs)


class ConvectiveCooling(ieee.ConvectiveCooling):
    """Convective cooling term.

    See ieee.ConvectiveCooling; it is exactly the same.
    """
