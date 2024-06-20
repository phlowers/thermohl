"""Power terms implementation matching cner's Excel sheet.

See NT-RD-CNER-DL-SLA-20-00215.
"""
from typing import Union

import numpy as np

from thermohl import air
from thermohl.power import ieee
from thermohl.power.base import PowerTerm


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
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.
        f : float or np.ndarray, optional
            Current frequency (Hz). The default is 50.

        """
        self.I = I
        self.D = D
        self.d = d
        self.kem = self._kem(A, a, km, ki)
        self.kl = kl
        self.kq = kq
        self.RDC20 = RDC20
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


class SolarHeating(ieee.SolarHeatingBase):
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
            **kwargs
    ):
        r"""Build with args.

        If more than one input are numpy arrays, they should have the same size.

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
        alpha : np.ndarray
            Solar absorption coefficient.
        srad : xxx
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        est = ieee._SRad(
            [-42., +63.8, -1.922, 0.03469, -3.61E-04, +1.943E-06, -4.08E-09],
            [0., 0., 0., 0., 0., 0., 0.]
        )
        for k in ['alt', 'tb']:
            if k in kwargs.keys():
                kwargs.pop(k)
        super().__init__(
            lat=lat, alt=0., azm=azm, tb=0., month=month, day=day, hour=hour, D=D, alpha=alpha, est=est, srad=srad,
            **kwargs
        )


class ConvectiveCooling(PowerTerm):
    """Convective cooling term.

    Very similar to IEEE. The differences are in some coefficient values for air
    constants.
    """

    def __init__(
            self,
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            Ta: Union[float, np.ndarray],
            ws: Union[float, np.ndarray],
            wa: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            **kwargs
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        alt : float or np.ndarray
            Altitude.
        azm : float or np.ndarray
            Azimuth.
        Ta : float or np.ndarray
            Ambient temperature.
        ws : float or np.ndarray
            Wind speed.
        wa : float or np.ndarray
            Wind angle regarding north.
        D : float or np.ndarray
            External diameter.

        """
        self.alt = alt
        self.Ta = Ta
        self.ws = ws
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))
        self.D = D

    def _value_forced(self, Tf, Td, vm):
        """Compute forced convective cooling value."""
        lf = air.IEEE.thermal_conductivity(Tf)
        # very slight difference with air.IEEE.dynamic_viscosity() due to the celsius/kelvin conversion
        mu = (1.458E-06 * (Tf + 273)**1.5) / (Tf + 383.4)
        Re = self.ws * self.D * vm / mu
        Kp = (1.194 - np.cos(self.da) + 0.194 * np.cos(2. * self.da) + 0.368 * np.sin(2. * self.da))
        return Kp * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6) * lf * Td

    def _value_natural(self, Td, vm):
        """Compute natural convective cooling value."""
        return 3.645 * np.sqrt(vm) * self.D**0.75 * np.sign(Td) * np.abs(Td)**1.25

    def value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute convective cooling.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Tf = 0.5 * (T + self.Ta)
        Td = T - self.Ta
        Tf = np.where(Tf < 0., 0., Tf)
        # very slight difference with air.IEEE.volumic_mass() in coefficient before alt**2
        vm = (1.293 - 1.525E-04 * self.alt + 6.38E-09 * self.alt**2) / (1 + 0.00367 * Tf)
        return np.maximum(self._value_forced(Tf, Td, vm), self._value_natural(Td, vm))


class RadiativeCooling(ieee.RadiativeCooling):
    pass
