"""Power terms implementation matching cner's Excel sheet.

See NT-RD-CNER-DL-SLA-20-00215.
"""
from typing import Union, Tuple

import numpy as np
from pyntb.optimize import fixed_point
from thermohl import air
from thermohl import olla
from thermohl import sun
from thermohl import utils


class JouleHeating(utils.PowerTerm):
    """Joule heating term."""

    @staticmethod
    def _temp_header(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                     D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                     A: Union[float, np.ndarray], a: Union[float, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray],
            Union[float, np.ndarray], Union[float, np.ndarray]]:
        d = d * np.ones_like(T)
        D = D * np.ones_like(T)
        a = a * np.ones_like(T)
        A = A * np.ones_like(T)
        c = 1 / 13 * np.ones_like(T)
        m = d > 0.
        c[m] = 1 / 21
        c *= I**2
        return c, d, D, a, A

    @staticmethod
    def _avg_core(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                  D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                  A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                  km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                  kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                  RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
                  f: Union[float, np.ndarray] = 50.,
                  xtol: float = 1.0E-03, maxiter: int = 128, **kwargs) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute average temperature and core temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a)

        def _Tmoy(x):
            return T + 0.5 * c * olla.JouleHeating._RAC(x, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)

        Ta = fixed_point(_Tmoy, np.zeros_like(T), xtol=xtol, maxiter=maxiter)
        Tc = 2. * Ta - T

        return Ta, Tc

    @staticmethod
    def _surf_core(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                   D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                   A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                   km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                   kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                   RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
                   f: Union[float, np.ndarray] = 50., **kwargs) \
            -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute surface temperature and core temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a)
        Ts = T - 0.5 * c * olla.JouleHeating._RAC(T, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
        Tc = 2. * T - Ts
        return Ts, Tc

    @staticmethod
    def _avg_surf(T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                  D: Union[float, np.ndarray], d: Union[float, np.ndarray],
                  A: Union[float, np.ndarray], a: Union[float, np.ndarray],
                  km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
                  kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
                  RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
                  f: Union[float, np.ndarray] = 50.,
                  xtol: float = 1.0E-03, maxiter: int = 128, **kwargs) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute average temperature and surface temperature."""
        c, d, D, a, A = JouleHeating._temp_header(T, I, D, d, A, a)

        def _Tmoy(x):
            return T - 0.5 * c * olla.JouleHeating._RAC(x, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)

        Ta = fixed_point(_Tmoy, np.zeros_like(T), xtol=xtol, maxiter=maxiter)
        Ts = 2. * Ta - T

        return Ta, Ts

    def value(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
              D: Union[float, np.ndarray], d: Union[float, np.ndarray],
              A: Union[float, np.ndarray], a: Union[float, np.ndarray],
              km: Union[float, np.ndarray], ki: Union[float, np.ndarray],
              kl: Union[float, np.ndarray], kq: Union[float, np.ndarray],
              RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
              f: Union[float, np.ndarray] = 50., **kwargs) -> Union[float, np.ndarray]:
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
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.
        f : float or np.ndarray, optional
            Current frequency (Hz). The default is 50.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Ta, _ = JouleHeating._avg_core(T, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
        Rac = olla.JouleHeating._RAC(Ta, I, D, d, A, a, km, ki, kl, kq, RDC20, T20, f)
        return Rac * I**2


class SolarHeating(utils.PowerTerm):
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
            -42. + 63.8 * x - 1.922 * x**2 + 0.03469 * x**3 - 3.61E-04 * x**4 + 1.943E-06 * x**5 - 4.08E-09 * x**6, 0.)

    @staticmethod
    def _solar_radiation(lat: Union[float, np.ndarray], azm: Union[float, np.ndarray],
                         month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]],
                         hour: Union[float, np.ndarray]) \
            -> Union[float, np.ndarray]:
        """Compute solar radiation.

        Difference with IEEE version are neither turbidity or altitude influence.
        """
        sa = sun.solar_altitude(lat, month, day, hour)
        sz = sun.solar_azimuth(lat, month, day, hour)
        th = np.arccos(np.cos(sa) * np.cos(sz - azm))
        Q = SolarHeating._catm(np.rad2deg(sa))
        return Q * np.sin(th)

    def value(self, T: Union[float, np.ndarray], lat: Union[float, np.ndarray], azm: Union[float, np.ndarray],
              month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]], hour: Union[float, np.ndarray],
              D: Union[float, np.ndarray], alpha: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        r"""Compute solar heating.

        If more than one input are numpy arrays, they should have the same size.

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
        alpha : np.ndarray
            Solar absorption coefficient.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        s = SolarHeating._solar_radiation(np.deg2rad(lat), np.deg2rad(azm), month, day, hour)
        return alpha * s * D * np.ones_like(T)

    def derivative(self, T: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class ConvectiveCooling(utils.PowerTerm):
    """Convective cooling term.

    Very similar to IEEE. The differences are in some coefficient values for air
    constants.
    """

    @staticmethod
    def _value_forced(Tf, Td, vm, ws, D, wa):
        """Compute forced convective cooling value."""
        lf = air.IEEE.thermal_conductivity(Tf)
        # very slight difference with air.IEEE.dynamic_viscosity() due to the
        # celsius/kelvin conversion
        mu = (1.458E-06 * (Tf + 273)**1.5) / (Tf + 383.4)
        Re = ws * D * vm / mu
        Kp = (1.194 - np.cos(wa) + 0.194 * np.cos(2. * wa) +
              0.368 * np.sin(2. * wa))
        return Kp * np.maximum(1.01 + 1.35 * Re**0.52, 0.754 * Re**0.6) * lf * Td

    @staticmethod
    def _value_natural(Td, vm, D):
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
        # very slight difference with air.IEEE.volumic_mass() in coefficient before alt**2
        vm = (1.293 - 1.525E-04 * alt + 6.38E-09 * alt**2) / (1 + 0.00367 * Tf)
        da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))
        return np.maximum(ConvectiveCooling._value_forced(Tf, Td, vm, ws, D, da),
                          ConvectiveCooling._value_natural(Td, vm, D))


class RadiativeCooling(utils.PowerTerm):
    """Power term for radiative cooling.

    Very similar to utils.RadiativeCooling. Difference are in the Stefan-Boltzman
    constant value and the celsius-kelvin conversion.
    """

    def value(self, T: Union[float, np.ndarray], Ta: Union[float, np.ndarray],
              D: Union[float, np.ndarray], epsilon: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return 1.78E-07 * epsilon * D * ((T + 273.)**4 - (Ta + 273.)**4)

    def derivative(self, T: Union[float, np.ndarray], Ta: Union[float, np.ndarray],
                   D: Union[float, np.ndarray], epsilon: Union[float, np.ndarray], **kwargs) -> Union[
        float, np.ndarray]:
        r"""Analytical derivative of value method.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return 4. * 1.78E-07 * epsilon * D * T**3
