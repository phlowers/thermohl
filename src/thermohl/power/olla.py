"""Power terms implementation using RTE's olla project choices.

Based on NT-RD-CNER-DL-SLA-20-00215 by RTE.
"""
from typing import Union

import numpy as np

from thermohl.power import ieee, cner
from thermohl.power.base import RadiativeCooling as RadiativeCooling_

_zerok = 273.15


def kelvin(t):
    return t + _zerok


class Air:
    """`Wikipedia <https://fr.wikipedia.org/wiki/Air> models."""

    @staticmethod
    def volumic_mass(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""
        Compute air volumic mass.

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
        Tk = kelvin(Tc)
        return 1.292 * _zerok * np.exp(-3.42E-02 * alt / Tk) / Tk

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
        Tk = kelvin(Tc)
        return 8.8848E-15 * Tk**3 - 3.2398E-11 * Tk**2 + 6.2657E-08 * Tk + 2.3543E-06

    @staticmethod
    def kinematic_viscosity(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[
        float, np.ndarray]:
        r"""Compute air kinematic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Kinematic viscosity in m\ :sup:`2`\ .s\ :sup:`-1`\ .

        """
        return Air.dynamic_viscosity(Tc) / Air.volumic_mass(Tc, alt=alt)

    @staticmethod
    def thermal_conductivity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air thermal conductivity.

        The output is valid for input in [-150, 1300] range (in Celsius)

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Thermal conductivity in W.m\ :sup:`-1`\ .K\ :sup:`-1`\ .

        """
        Tk = kelvin(Tc)
        return (1.5207E-11 * Tk**3 - 4.8570E-08 * Tk**2 + 1.0184E-04 * Tk - 3.9333E-04)


class JouleHeating(cner.JouleHeating):
    """Joule heating term."""
    pass


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def __init__(
            self,
            lat: Union[float, np.ndarray],
            alt: Union[float, np.ndarray],
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
        alt : float or np.ndarray
             Lltitude.
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
        if 'tb' in kwargs.keys():
            kwargs.pop('tb')
        super().__init__(
            lat=lat, alt=alt, azm=azm, tb=0., month=month, day=day, hour=hour, D=D, alpha=alpha, srad=srad, **kwargs
        )


class ConvectiveCooling(ieee.ConvectiveCoolingBase):
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
        super().__init__(
            alt, azm, Ta, ws, wa, D, Air.volumic_mass, Air.dynamic_viscosity, Air.thermal_conductivity, **kwargs
        )


class RadiativeCooling(RadiativeCooling_):
    pass
