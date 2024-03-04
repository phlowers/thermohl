"""Power terms implementation using CIGRE recommendations.

See Thermal behaviour of overhead conductors, study committee 22, working
group 12, 2002.
"""
from typing import Union

import numpy as np
import thermohl.air as air
import thermohl.sun as sun
from thermohl.utils import PowerTerm
from thermohl.utils import RadiativeCooling as RCu


class JouleHeating(PowerTerm):
    """Joule heating term."""

    def value(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
              km: Union[float, np.ndarray], kl: Union[float, np.ndarray],
              RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
              **kwargs) -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
        I : float or np.ndarray
            Transit intensity.
        km : float or np.ndarray
            Coefficient for magnetic effects.
        kl : float or np.ndarray
            Linear resistance augmentation with temperature.
        RDC20 : float or np.ndarray
            Electric resistance per unit length (DC) at 20°C.
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return km * RDC20 * (1. + kl * (T - T20)) * I**2

    def derivative(self, T: Union[float, np.ndarray], I: Union[float, np.ndarray],
                   km: Union[float, np.ndarray], kl: Union[float, np.ndarray],
                   RDC20: Union[float, np.ndarray], T20: Union[float, np.ndarray] = 20.,
                   **kwargs) -> Union[float, np.ndarray]:
        r"""Compute joule heating derivative.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
        I : float or np.ndarray
            Transit intensity.
        km : float or np.ndarray
            Coefficient for magnetic effects.
        kl : float or np.ndarray
            Linear resistance augmentation with temperature.
        RDC20 : float or np.ndarray
            Electric resistance per unit length (DC) at 20°C.
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return km * RDC20 * kl * I**2 * np.ones_like(T)


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _solar_radiation(lat: Union[float, np.ndarray], azm: Union[float, np.ndarray], albedo: Union[float, np.ndarray],
                         month: Union[int, np.ndarray[int]], day: Union[int, np.ndarray[int]], hour) -> np.ndarray:
        """Compute solar radiation."""
        sd = sun.solar_declination(month, day)
        sh = sun.hour_angle(hour)
        sa = sun.solar_altitude(lat, month, day, hour)
        Id = 1280. * np.sin(sa) / (0.314 + np.sin(sa))
        gs = np.arcsin(np.cos(sd) * np.sin(sh) / np.cos(sa))
        eta = np.arccos(np.cos(sa) * np.cos(gs - azm))
        A = 0.5 * np.pi * albedo * np.sin(sa) + np.sin(eta)
        x = np.sin(sa)
        C = np.piecewise(x, [x < 0., x >= 0.], [lambda x_: 0., lambda x_: x_**1.2])
        B = 0.5 * np.pi * (1 + albedo) * (570. - 0.47 * Id) * C
        return np.where(sa > 0., A * Id + B, 0.)

    def value(self, T: Union[float, np.ndarray],
              lat: Union[float, np.ndarray], azm: Union[float, np.ndarray], al: Union[float, np.ndarray],
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
        azm : float or np.ndarray
            Azimuth.
        al : float or np.ndarray
            Albedo.
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
        s = SolarHeating._solar_radiation(np.deg2rad(lat), np.deg2rad(azm), al, month, day, hour)
        return alpha * s * D * np.ones_like(T)

    def derivative(self, T: Union[float, np.ndarray], **kwargs) -> np.ndarray:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class ConvectiveCooling(PowerTerm):
    """Convective cooling term."""

    @staticmethod
    def _nu_forced(Tf: Union[float, np.ndarray], nu: Union[float, np.ndarray], alt: Union[float, np.ndarray],
                   ws: Union[float, np.ndarray], wa: Union[float, np.ndarray],
                   D: Union[float, np.ndarray], R: Union[float, np.ndarray]) -> np.ndarray:
        """Nusselt number for forced convection."""
        rd = air.CIGRE.relative_density(Tf, alt)
        Re = rd * np.abs(ws) * D / nu

        B1 = 0.048 * np.ones_like(Re)
        B1 = np.where(R < 0.05, 0.178, B1)
        B1 = np.where(Re <= 2.65E+03, 0.641, B1)

        n = 0.800 * np.ones_like(Re)
        n = np.where(R < 0.05, 0.633, n)
        n = np.where(Re <= 2.65E+03, 0.471, n)

        B2 = np.where(wa < np.deg2rad(24.), 0.68, 0.58)
        m1 = np.where(wa < np.deg2rad(24.), 1.08, 0.90)

        return np.maximum(0.42 + B2 * np.sin(wa)**m1, 0.55) * (B1 * Re**n)

    @staticmethod
    def _nu_natural(Tf: Union[float, np.ndarray], Td: Union[float, np.ndarray], nu: Union[float, np.ndarray],
                    D: Union[float, np.ndarray], g: Union[float, np.ndarray]) -> np.ndarray:
        """Nusselt number for natural convection."""
        gr = D**3 * np.abs(Td) * g / ((Tf + 273.15) * nu**2)
        gp = gr * air.CIGRE.prandtl(Tf)
        # gp[gp < 0.] = 0.
        ia = gp < 1.0E+04
        A2, m2 = (np.zeros_like(Tf),) * 2
        A2[ia] = 0.850
        m2[ia] = 0.188
        A2[~ia] = 0.480
        m2[~ia] = 0.250
        return A2 * gp**m2

    def value(self, T: Union[float, np.ndarray], alt: Union[float, np.ndarray], azm: Union[float, np.ndarray],
              Ta: Union[float, np.ndarray], ws: Union[float, np.ndarray], wa: Union[float, np.ndarray],
              D: Union[float, np.ndarray], R: Union[float, np.ndarray], g: float = 9.81, **kwargs) -> Union[
        float, np.ndarray]:
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
        R : float or np.ndarray
            Cable roughness.
        g : float, optional
            Gravitational acceleration. The default is 9.81.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Tf = 0.5 * (T + Ta)
        Td = T - Ta
        lm = air.CIGRE.thermal_conductivity(Tf)
        # lm[lm < 0.01] = 0.01
        nu = air.CIGRE.kinematic_viscosity(Tf)
        # nu[nu < 1.0E-06] = 1.0E-06
        da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))
        nf = ConvectiveCooling._nu_forced(Tf, nu, alt, ws, da, D, R)
        nn = ConvectiveCooling._nu_natural(Tf, Td, nu, D, g)
        return np.pi * lm * (T - Ta) * np.maximum(nf, nn)


class RadiativeCooling(RCu):
    """."""
