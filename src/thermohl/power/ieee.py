"""Power terms implementation using IEEE std 38-2012 models.

IEEE std 38-2012 is the IEEE Standard for Calculating the Current-Temperature
Relationship of Bare Overhead Conductors.
"""
from typing import Union

import numpy as np

import thermohl.sun as sun
from thermohl.power.base import PowerTerm


class Air:
    """Air quantities."""

    @staticmethod
    def volumic_mass(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
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
        return (1.293 - 1.525E-04 * alt + 6.379E-09 * alt**2) / (1. + 0.00367 * Tc)

    @staticmethod
    def dynamic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        return (1.458E-06 * (Tc + 273.)**1.5) / (Tc + 383.4)

    @staticmethod
    def thermal_conductivity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        return 2.424E-02 + 7.477E-05 * Tc - 4.407E-09 * Tc**2


class JouleHeating(PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _c(TLow, THigh, RDCLow, RDCHigh):
        return (RDCHigh - RDCLow) / (THigh - TLow)

    def __init__(
            self,
            I: Union[float, np.ndarray],
            TLow: Union[float, np.ndarray],
            THigh: Union[float, np.ndarray],
            RDCLow: Union[float, np.ndarray],
            RDCHigh: Union[float, np.ndarray],
            **kwargs
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
        return (self.RDCLow + self.c * (T - self.TLow)) * self.I**2

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

    def __init__(self, cl, il):
        self.clean = cl
        self.indus = il

    def catm(self, x: Union[float, np.ndarray], trb: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute coefficient for atmosphere turbidity."""
        omt = 1. - trb
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
            lat: Union[float, np.ndarray],
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            trb: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Compute solar radiation."""
        sa = sun.solar_altitude(lat, month, day, hour)
        sz = sun.solar_azimuth(lat, month, day, hour)
        th = np.arccos(np.cos(sa) * np.cos(sz - azm))
        K = 1. + 1.148E-04 * alt - 1.108E-08 * alt**2
        Q = self.catm(np.rad2deg(sa), trb)
        sr = K * Q * np.sin(th)
        return np.where(sr > 0., sr, 0.)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
            self,
            lat: Union[float, np.ndarray],
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            tb: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray],
            est: _SRad,
            srad: Union[float, np.ndarray] = None,
            **kwargs,
    ):
        self.alpha = alpha
        if srad is None:
            self.srad = est(np.deg2rad(lat), alt, np.deg2rad(azm), tb, month, day, hour)
        else:
            self.srad = srad
        self.D = D

    def value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class SolarHeating(SolarHeatingBase):
    def __init__(
            self,
            lat: Union[float, np.ndarray],
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            tb: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray],
            srad: Union[float, np.ndarray] = None,
            **kwargs,
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
            [-4.22391E+01, +6.38044E+01, -1.9220E+00, +3.46921E-02, -3.61118E-04, +1.94318E-06, -4.07608E-09],
            [+5.31821E+01, +1.4211E+01, +6.6138E-01, -3.1658E-02, +5.4654E-04, -4.3446E-06, +1.3236E-08]
        )
        super().__init__(lat, alt, azm, tb, month, day, hour, D, alpha, est, srad, **kwargs)


class ConvectiveCoolingBase(PowerTerm):
    """Convective cooling term."""

    def __init__(
            self,
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            Ta: Union[float, np.ndarray],
            ws: Union[float, np.ndarray],
            wa: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            rho, mu, lambda_,
            **kwargs,
    ):
        self.alt = alt
        self.Ta = Ta
        self.ws = ws
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))
        self.D = D

        self.rho = rho
        self.mu = mu
        self.lambda_ = lambda_

    def _value_forced(
            self,
            Tf: Union[float, np.ndarray],
            Td: Union[float, np.ndarray],
            vm: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute forced convective cooling value."""
        Re = self.ws * self.D * vm / self.mu(Tf)
        Kp = (1.194 - np.cos(self.da) + 0.194 * np.cos(2. * self.da) + 0.368 * np.sin(2. * self.da))
        return Kp * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6) * self.lambda_(Tf) * Td

    def _value_natural(
            self,
            Td: Union[float, np.ndarray],
            vm: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
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
        vm = self.rho(Tf, self.alt)
        return np.maximum(
            self._value_forced(Tf, Td, vm),
            self._value_natural(Td, vm)
        )


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term."""

    def __init__(
            self,
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            Ta: Union[float, np.ndarray],
            ws: Union[float, np.ndarray],
            wa: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            **kwargs,
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
        super().__init__(alt, azm, Ta, ws, wa, D, Air.volumic_mass, Air.dynamic_viscosity, Air.thermal_conductivity)


class RadiativeCooling(PowerTerm):
    """Power term for radiative cooling.

    Very similar to thermohl.power.base.RadiativeCooling. Difference are in the
    Stefan-Boltzman constant value and the celsius-kelvin conversion.
    """

    def __init__(
            self,
            Ta: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            epsilon: Union[float, np.ndarray],
            **kwargs
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

    def value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        return 17.8 * self.epsilon * self.D * (((T + 273.) / 100.)**4 - ((self.Ta + 273.) / 100.)**4)

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        return 4. * 1.78E-07 * self.epsilon * self.D * T**3
