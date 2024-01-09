"""Power terms implementation using IEEE std 38-2012 models.

IEEE std 38-2012 is the IEEE Standard for Calculating the Current-Temperature
Relationship of Bare Overhead Conductors.
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union

import thermohl.air as air
import thermohl.sun as sun
from thermohl.utils import PowerTerm
from thermohl.utils import RadiativeCooling


class JouleHeating(PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _c(TLow, THigh, RDCLow, RDCHigh):
        return (RDCHigh - RDCLow) / (THigh - TLow)

    def value(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
              TLow: Union[float, np.ndarray], THigh: Union[float, np.ndarray],
              RDCLow: Union[float, np.ndarray], RDCHigh: Union[float, np.ndarray], **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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
        c = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)
        return (RDCLow + c * (T - TLow)) * I**2

    def derivative(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                   TLow: Union[float, np.ndarray], THigh: Union[float, np.ndarray],
                   RDCLow: Union[float, np.ndarray], RDCHigh: Union[float, np.ndarray], **kwargs) \
            -> Union[float, np.ndarray]:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        c = JouleHeating._c(TLow, THigh, RDCLow, RDCHigh)
        return c * I**2 * np.ones_like(T)


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _catm(x: Union[float, np.ndarray], trb: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute coefficient for atmosphere turbidity."""
        clean = np.array(
            [-4.22391E+01, +6.38044E+01, -1.9220E+00, +3.46921E-02, -3.61118E-04, +1.94318E-06, -4.07608E-09])
        indus = np.array(
            [+5.31821E+01, +1.4211E+01, +6.6138E-01, -3.1658E-02, +5.4654E-04, -4.3446E-06, +1.3236E-08])
        A = (1. - trb) * clean[6] + trb * indus[6]
        B = (1. - trb) * clean[5] + trb * indus[5]
        C = (1. - trb) * clean[4] + trb * indus[4]
        D = (1. - trb) * clean[3] + trb * indus[3]
        E = (1. - trb) * clean[2] + trb * indus[2]
        F = (1. - trb) * clean[1] + trb * indus[1]
        G = (1. - trb) * clean[0] + trb * indus[0]
        return A * x**6 + B * x**5 + C * x**4 + D * x**3 + E * x**2 + F * x**1 + G

    @staticmethod
    def _solar_radiation(lat: Union[float, np.ndarray], alt: Union[float, np.ndarray], azm: Union[float, np.ndarray],
                         trb: Union[float, np.ndarray],
                         month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]], hour: Union[float, np.ndarray]) \
            -> np.ndarray:
        """Compute solar radiation."""
        sa = sun.solar_altitude(lat, month, day, hour)
        sz = sun.solar_azimuth(lat, month, day, hour)
        th = np.arccos(np.cos(sa) * np.cos(sz - azm))
        K = 1. + 1.148E-04 * alt - 1.108E-08 * alt**2
        Q = SolarHeating._catm(np.rad2deg(sa), trb)
        sr = K * Q * np.sin(th)
        return np.where(sr > 0., sr, 0.)

    def value(self, T: Union[float, np.ndarray],
              lat: Union[float, np.ndarray], alt: Union[float, np.ndarray], azm: Union[float, np.ndarray],
              tb: Union[float, np.ndarray],
              month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]], hour: Union[float, np.ndarray],
              D: Union[float, np.ndarray], alpha: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        r"""Compute solar heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
        lat : float or np.ndarray
            Latitude.
        alt : float or np.ndarray
            altitude.
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

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        s = SolarHeating._solar_radiation(np.deg2rad(lat), alt, np.deg2rad(azm), tb, month, day, hour)
        return alpha * s * D * np.ones_like(T)

    def derivative(self, T: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class ConvectiveCooling(PowerTerm):
    """Convective cooling term."""

    @staticmethod
    def _value_forced(Tf: Union[float, np.ndarray], Td: Union[float, np.ndarray],
                      vm: Union[float, np.ndarray], ws: Union[float, np.ndarray],
                      D: Union[float, np.ndarray], wa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute forced convective cooling value."""
        lf = air.IEEE.thermal_conductivity(Tf)
        mu = air.IEEE.dynamic_viscosity(Tf)
        Re = ws * D * vm / mu
        Kp = (1.194 - np.cos(wa) + 0.194 * np.cos(2. * wa) +
              0.368 * np.sin(2. * wa))
        return Kp * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6) * lf * Td

    @staticmethod
    def _value_natural(Td: Union[float, np.ndarray], vm: Union[float, np.ndarray],
                       D: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute natural convective cooling value."""
        return 3.645 * np.sqrt(vm) * D**0.75 * np.sign(Td) * np.abs(Td)**1.25

    def value(self, T: Union[float, np.ndarray], alt: Union[float, np.ndarray], azm: Union[float, np.ndarray],
              Ta: Union[float, np.ndarray], ws: Union[float, np.ndarray], wa: Union[float, np.ndarray],
              D: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        r"""Compute convective cooling.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Tf = 0.5 * (T + Ta)
        Td = T - Ta
        Tf[Tf < 0.] = 0.
        vm = air.IEEE.volumic_mass(Tf, alt)
        da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))
        return np.maximum(ConvectiveCooling._value_forced(Tf, Td, vm, ws, D, da),
                          ConvectiveCooling._value_natural(Td, vm, D))
