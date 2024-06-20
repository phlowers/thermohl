"""Power terms implementation matching cner's Excel sheet.

See NT-RD-CNER-DL-SLA-20-00215.
"""
from typing import Union

import numpy as np

from thermohl import air
from thermohl import sun
from thermohl.power import olla
from thermohl.power.base import PowerTerm


class JouleHeating(olla.JouleHeating):
    """Joule heating term."""
    pass


class SolarHeating(PowerTerm):
    """Solar heating term.

    Very similar to IEEE. Differences explained in methods' comments.
    """

    @staticmethod
    def _catm(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute coefficient for atmosphere turbidity.

        Differences with IEEE version is clean air only and slightly different
        coefficients.
        """
        return np.maximum(
            -42. + 63.8 * x - 1.922 * x**2 + 0.03469 * x**3 - 3.61E-04 * x**4 + 1.943E-06 * x**5 - 4.08E-09 * x**6,
            0.
        )

    @staticmethod
    def _solar_radiation(
            lat: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute solar radiation.

        Difference with IEEE version are neither turbidity or altitude influence.
        """
        sa = sun.solar_altitude(lat, month, day, hour)
        sz = sun.solar_azimuth(lat, month, day, hour)
        th = np.arccos(np.cos(sa) * np.cos(sz - azm))
        Q = SolarHeating._catm(np.rad2deg(sa))
        return Q * np.sin(th)

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
        self.alpha = alpha
        if srad is None:
            self.srad = SolarHeating._solar_radiation(np.deg2rad(lat), np.deg2rad(azm), month, day, hour)
        else:
            self.srad = srad
        self.D = D

    def value(self, T: Union[float, np.ndarray]) -> np.ndarray:
        r"""Compute solar heating.

        If more than one input are numpy arrays, they should have the same size.

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

    def derivative(self, T: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


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
        return 1.78E-07 * self.epsilon * self.D * ((T + 273.)**4 - (self.Ta + 273.)**4)

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
