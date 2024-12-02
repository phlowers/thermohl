"""Power terms implementation using IEEE std 38-2012 models.

IEEE std 38-2012 is the IEEE Standard for Calculating the Current-Temperature
Relationship of Bare Overhead Conductors.
"""

from typing import Any, Optional, List, Callable

import numpy as np

import thermohl.sun as sun
from thermohl import floatArrayLike, intArrayLike
from thermohl.power.base import PowerTerm


class Air:
    """Air quantities."""

    @staticmethod
    def volumic_mass(Tc: floatArrayLike, alt: floatArrayLike = 0.0) -> floatArrayLike:
        r"""Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Volumic mass in kg.m\ :sup:`-3`\ .

        """
        return (1.293 - 1.525e-04 * alt + 6.379e-09 * alt**2) / (1.0 + 0.00367 * Tc)

    @staticmethod
    def dynamic_viscosity(Tc: floatArrayLike) -> floatArrayLike:
        r"""Compute air dynamic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

        """
        return (1.458e-06 * (Tc + 273.0) ** 1.5) / (Tc + 383.4)

    @staticmethod
    def thermal_conductivity(Tc: floatArrayLike) -> floatArrayLike:
        r"""Compute air thermal conductivity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Thermal conductivity in W.m\ :sup:`-1`\ .K\ :sup:`-1`\ .

        """
        return 2.424e-02 + 7.477e-05 * Tc - 4.407e-09 * Tc**2


class JouleHeating(PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _c(
        TLow: floatArrayLike,
        THigh: floatArrayLike,
        RDCLow: floatArrayLike,
        RDCHigh: floatArrayLike,
    ) -> floatArrayLike:
        return (RDCHigh - RDCLow) / (THigh - TLow)

    def __init__(
        self,
        I: floatArrayLike,
        TLow: floatArrayLike,
        THigh: floatArrayLike,
        RDCLow: floatArrayLike,
        RDCHigh: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        I : float or np.ndarray
            Transit intensity.
        TLow : float or np.ndarray
            Temperature for RDCHigh measurement.
        THigh : float or np.ndarray
            Temperature for RDCHigh measurement.
        RDCLow : float or np.ndarray
            Electric resistance per unit length at TLow .
        RDCHigh : float or np.ndarray
            Electric resistance per unit length at THigh.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        self.TLow = TLow
        self.THigh = THigh
        self.RDCLow = RDCLow
        self.RDCHigh = RDCHigh
        self.I = I
        self.c = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)

    def _rdc(self, T: floatArrayLike) -> floatArrayLike:
        return self.RDCLow + self.c * (T - self.TLow)

    def value(self, T: floatArrayLike) -> floatArrayLike:
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
        return self._rdc(T) * self.I**2

    def derivative(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return self.c * self.I**2 * np.ones_like(T)


class _SRad:

    def __init__(self, clean: List[float], indus: List[float]):
        self.clean = clean
        self.indus = indus

    def catm(self, x: floatArrayLike, trb: floatArrayLike) -> floatArrayLike:
        """Compute coefficient for atmosphere turbidity."""
        omt = 1.0 - trb
        A = omt * self.clean[6] + trb * self.indus[6]
        B = omt * self.clean[5] + trb * self.indus[5]
        C = omt * self.clean[4] + trb * self.indus[4]
        D = omt * self.clean[3] + trb * self.indus[3]
        E = omt * self.clean[2] + trb * self.indus[2]
        F = omt * self.clean[1] + trb * self.indus[1]
        G = omt * self.clean[0] + trb * self.indus[0]
        return A * x**6 + B * x**5 + C * x**4 + D * x**3 + E * x**2 + F * x**1 + G

    def __call__(
        self,
        lat: floatArrayLike,
        alt: floatArrayLike,
        azm: floatArrayLike,
        trb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        sa = sun.solar_altitude(lat, month, day, hour)
        sz = sun.solar_azimuth(lat, month, day, hour)
        th = np.arccos(np.cos(sa) * np.cos(sz - azm))
        K = 1.0 + 1.148e-04 * alt - 1.108e-08 * alt**2
        Q = self.catm(np.rad2deg(sa), trb)
        sr = K * Q * np.sin(th)
        return np.where(sr > 0.0, sr, 0.0)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
        self,
        lat: floatArrayLike,
        alt: floatArrayLike,
        azm: floatArrayLike,
        tb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        D: floatArrayLike,
        alpha: floatArrayLike,
        est: _SRad,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        self.alpha = alpha
        if srad is None:
            self.srad = est(np.deg2rad(lat), alt, np.deg2rad(azm), tb, month, day, hour)
        else:
            self.srad = np.maximum(srad, 0.0)
        self.D = D

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute solar heating.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return self.alpha * self.srad * self.D * np.ones_like(T)

    def derivative(self, T: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        lat: floatArrayLike,
        alt: floatArrayLike,
        azm: floatArrayLike,
        tb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        D: floatArrayLike,
        alpha: floatArrayLike,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude.
        alt : float or np.ndarray
            Altitude.
        azm : float or np.ndarray
            Azimuth.
        tb : float or np.ndarray
            Air pollution from 0 (clean) to 1 (polluted).
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
        est = _SRad(
            [
                -4.22391e01,
                +6.38044e01,
                -1.9220e00,
                +3.46921e-02,
                -3.61118e-04,
                +1.94318e-06,
                -4.07608e-09,
            ],
            [
                +5.31821e01,
                +1.4211e01,
                +6.6138e-01,
                -3.1658e-02,
                +5.4654e-04,
                -4.3446e-06,
                +1.3236e-08,
            ],
        )
        super().__init__(
            lat, alt, azm, tb, month, day, hour, D, alpha, est, srad, **kwargs
        )


class ConvectiveCoolingBase(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        rho: Callable[[floatArrayLike, floatArrayLike], floatArrayLike],
        mu: Callable[[floatArrayLike], floatArrayLike],
        lambda_: Callable[[floatArrayLike], floatArrayLike],
        **kwargs: Any,
    ):
        self.alt = alt
        self.Ta = Ta
        self.ws = ws
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.0)))
        self.D = D

        self.rho = rho
        self.mu = mu
        self.lambda_ = lambda_

    def _value_forced(
        self,
        Tf: floatArrayLike,
        Td: floatArrayLike,
        vm: floatArrayLike,
    ) -> floatArrayLike:
        """Compute forced convective cooling value."""
        Re = self.ws * self.D * vm / self.mu(Tf)
        Kp = (
            1.194
            - np.cos(self.da)
            + 0.194 * np.cos(2.0 * self.da)
            + 0.368 * np.sin(2.0 * self.da)
        )
        return (
            Kp
            * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6)
            * self.lambda_(Tf)
            * Td
        )

    def _value_natural(
        self,
        Td: floatArrayLike,
        vm: floatArrayLike,
    ) -> floatArrayLike:
        """Compute natural convective cooling value."""
        return 3.645 * np.sqrt(vm) * self.D**0.75 * np.sign(Td) * np.abs(Td) ** 1.25

    def value(self, T: floatArrayLike) -> floatArrayLike:
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
        vm = self.rho(Tf, self.alt)
        return np.maximum(self._value_forced(Tf, Td, vm), self._value_natural(Td, vm))


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term."""

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        **kwargs: Any,
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
        super().__init__(
            alt,
            azm,
            Ta,
            ws,
            wa,
            D,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
        )


class RadiativeCooling(PowerTerm):
    """Power term for radiative cooling.

    Very similar to thermohl.power.base.RadiativeCooling. Difference are in the
    Stefan-Boltzman constant value and the celsius-kelvin conversion.
    """

    def __init__(
        self,
        Ta: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        Parameters
        ----------
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.

        """
        self.Ta = Ta
        self.D = D
        self.epsilon = epsilon

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return (
            17.8
            * self.epsilon
            * self.D
            * (((T + 273.0) / 100.0) ** 4 - ((self.Ta + 273.0) / 100.0) ** 4)
        )

    def derivative(self, T: floatArrayLike) -> floatArrayLike:
        r"""Analytical derivative of value method.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return 4.0 * 1.78e-07 * self.epsilon * self.D * T**3
