"""Power terms implementation using CIGRE recommendations.

See Thermal behaviour of overhead conductors, study committee 22, working
group 12, 2002.
"""
from typing import Union

import numpy as np

import thermohl.sun as sun
from thermohl.power.base import PowerTerm
from thermohl.power.base import RadiativeCooling as RadiativeCooling_


class Air:
    """CIGRE air models."""

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
        return 1.2925 * Air.relative_density(Tc, alt)

    @staticmethod
    def relative_density(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        """Compute relative density, ie density ratio regarding density at zero altitude.

        This function has temperature and altitude as input for consistency
        regarding other functions in the module, but the temperature has no
        influence, only the altitude for this model.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.exp(-1.16E-04 * alt) * np.ones_like(Tc)

    @staticmethod
    def kinematic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air kinematic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Kinematic viscosity in m\ :sup:`2`\ .s\ :sup:`-1`\ .

        """
        return 1.32E-05 + 9.5E-08 * Tc

    @staticmethod
    def dynamic_viscosity(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""Compute air dynamic viscosity.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

        """
        return Air.kinematic_viscosity(Tc) * Air.volumic_mass(Tc, alt)

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
        return 2.42E-02 + 7.2E-05 * Tc

    @staticmethod
    def prandtl(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Prandtl number.

        The Prandtl number (Pr) is a dimensionless number, named after the German
        physicist Ludwig Prandtl, defined as the ratio of momentum diffusivity to
        thermal diffusivity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Prandtl number (no unit)

        """
        return 0.715 - 2.5E-04 * Tc


class JouleHeating(PowerTerm):
    """Joule heating term."""

    def __init__(
            self,
            I: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            T20: Union[float, np.ndarray] = 20.,
            **kwargs,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
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

        """
        self.I = I
        self.km = km
        self.kl = kl
        self.RDC20 = RDC20
        self.T20 = T20

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
        return self.km * self.RDC20 * (1. + self.kl * (T - self.T20)) * self.I**2

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
        return self.km * self.RDC20 * self.kl * self.I**2 * np.ones_like(T)


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _solar_radiation(
            lat: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            albedo: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int,
            np.ndarray[int]], hour
    ) -> np.ndarray:
        """Compute solar radiation."""
        sd = sun.solar_declination(month, day)
        sh = sun.hour_angle(hour)
        sa = sun.solar_altitude(lat, month, day, hour)
        Id = 1280. * np.sin(sa) / (0.314 + np.sin(sa))
        gs = np.arcsin(np.cos(sd) * np.sin(sh) / np.cos(sa))
        eta = np.arccos(np.cos(sa) * np.cos(gs - azm))
        A = 0.5 * np.pi * albedo * np.sin(sa) + np.sin(eta)
        x = np.sin(sa)
        C = np.piecewise(
            x,
            [x < 0., x >= 0.],
            [lambda x_: 0., lambda x_: x_**1.2]
        )
        B = 0.5 * np.pi * (1 + albedo) * (570. - 0.47 * Id) * C
        return np.where(sa > 0., A * Id + B, 0.)

    def __init__(
            self,
            lat: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            al: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray],
            srad: Union[float, np.ndarray] = None,
            **kwargs
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
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
        srad : xxx
            xxx.


        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        self.alpha = alpha
        if srad is None:
            self.srad = SolarHeating._solar_radiation(np.deg2rad(lat), np.deg2rad(azm), al, month, day, hour)
        else:
            self.srad = srad
        self.D = D

    def value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
    """Convective cooling term."""

    def __init__(
            self,
            alt: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            Ta: Union[float, np.ndarray],
            ws: Union[float, np.ndarray],
            wa: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            R: Union[float, np.ndarray],
            g: float = 9.81,
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
        R : float or np.ndarray
            Cable roughness.
        g : float, optional
            Gravitational acceleration. The default is 9.81.

        """
        self.alt = alt
        self.Ta = Ta
        self.ws = ws
        self.D = D
        self.R = R
        self.g = g
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.)))

    def _nu_forced(self, Tf: Union[float, np.ndarray], nu: Union[float, np.ndarray]) -> np.ndarray:
        """Nusselt number for forced convection."""
        rd = Air.relative_density(Tf, self.alt)
        Re = rd * np.abs(self.ws) * self.D / nu

        B1 = 0.048 * np.ones_like(Re)
        B1 = np.where(self.R < 0.05, 0.178, B1)
        B1 = np.where(Re <= 2.65E+03, 0.641, B1)

        n = 0.800 * np.ones_like(Re)
        n = np.where(self.R < 0.05, 0.633, n)
        n = np.where(Re <= 2.65E+03, 0.471, n)

        B2 = np.where(self.da < np.deg2rad(24.), 0.68, 0.58)
        m1 = np.where(self.da < np.deg2rad(24.), 1.08, 0.90)

        return np.maximum(0.42 + B2 * np.sin(self.da)**m1, 0.55) * (B1 * Re**n)

    def _nu_natural(
            self,
            Tf: Union[float, np.ndarray],
            Td: Union[float, np.ndarray],
            nu: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Nusselt number for natural convection."""
        gr = self.D**3 * np.abs(Td) * self.g / ((Tf + 273.15) * nu**2)
        gp = gr * Air.prandtl(Tf)
        # gp[gp < 0.] = 0.
        ia = gp < 1.0E+04
        A2, m2 = (np.zeros_like(Tf),) * 2
        A2[ia] = 0.850
        m2[ia] = 0.188
        A2[~ia] = 0.480
        m2[~ia] = 0.250
        return A2 * gp**m2

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
        nu = Air.kinematic_viscosity(Tf)
        # nu[nu < 1.0E-06] = 1.0E-06
        lm = Air.thermal_conductivity(Tf)
        # lm[lm < 0.01] = 0.01
        nf = self._nu_forced(Tf, nu)
        nn = self._nu_natural(Tf, Td, nu)
        return np.pi * lm * (T - self.Ta) * np.maximum(nf, nn)


class RadiativeCooling(RadiativeCooling_):

    def __init__(
            self,
            Ta: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            epsilon: Union[float, np.ndarray],
            sigma: float = 5.67E-08,
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
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.

        Returns
        -------

        """
        super().__init__(Ta=Ta, D=D, epsilon=epsilon, sigma=sigma, zerok=273., **kwargs)
