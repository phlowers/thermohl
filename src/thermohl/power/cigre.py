"""Power terms implementation using CIGRE recommendations.

See Thermal behaviour of overhead conductors, study committee 22, working
group 12, 2002.
"""

from typing import Any, Optional

import numpy as np

import thermohl.sun as sun
from thermohl import floatArrayLike, intArrayLike
from thermohl.power.base import PowerTerm
from thermohl.power.base import RadiativeCooling as RadiativeCooling_


class Air:
    """CIGRE air models."""

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
        return 1.2925 * Air.relative_density(Tc, alt)

    @staticmethod
    def relative_density(
        Tc: floatArrayLike, alt: floatArrayLike = 0.0
    ) -> floatArrayLike:
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
        return np.exp(-1.16e-04 * alt) * np.ones_like(Tc)

    @staticmethod
    def kinematic_viscosity(Tc: floatArrayLike) -> floatArrayLike:
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
        return 1.32e-05 + 9.5e-08 * Tc

    @staticmethod
    def dynamic_viscosity(
        Tc: floatArrayLike, alt: floatArrayLike = 0.0
    ) -> floatArrayLike:
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
        return 2.42e-02 + 7.2e-05 * Tc

    @staticmethod
    def prandtl(Tc: floatArrayLike) -> floatArrayLike:
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
        return 0.715 - 2.5e-04 * Tc


class JouleHeating(PowerTerm):
    """Joule heating term."""

    def __init__(
        self,
        I: floatArrayLike,
        km: floatArrayLike,
        kl: floatArrayLike,
        RDC20: floatArrayLike,
        T20: floatArrayLike = 20.0,
        **kwargs: Any,
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
        return self.km * self.RDC20 * (1.0 + self.kl * (T - self.T20)) * self.I**2

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
        return self.km * self.RDC20 * self.kl * self.I**2 * np.ones_like(T)


class SolarHeating(PowerTerm):
    """Solar heating term."""

    @staticmethod
    def _solar_radiation(
        lat: floatArrayLike,
        azm: floatArrayLike,
        albedo: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        sd = sun.solar_declination(month, day)
        sh = sun.hour_angle(hour)
        sa = sun.solar_altitude(lat, month, day, hour)
        Id = 1280.0 * np.sin(sa) / (0.314 + np.sin(sa))
        gs = np.arcsin(np.cos(sd) * np.sin(sh) / np.cos(sa))
        eta = np.arccos(np.cos(sa) * np.cos(gs - azm))
        A = 0.5 * np.pi * albedo * np.sin(sa) + np.sin(eta)
        x = np.sin(sa)
        C = np.piecewise(x, [x < 0.0, x >= 0.0], [lambda x_: 0.0, lambda x_: x_**1.2])
        B = 0.5 * np.pi * (1 + albedo) * (570.0 - 0.47 * Id) * C
        return np.where(sa > 0.0, A * Id + B, 0.0)

    def __init__(
        self,
        lat: floatArrayLike,
        azm: floatArrayLike,
        al: floatArrayLike,
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
            self.srad = SolarHeating._solar_radiation(
                np.deg2rad(lat), np.deg2rad(azm), al, month, day, hour
            )
        else:
            self.srad = srad
        self.D = D

    def value(self, T: floatArrayLike) -> floatArrayLike:
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

    def derivative(self, T: floatArrayLike, **kwargs: Any) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(T)


class ConvectiveCooling(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        R: floatArrayLike,
        g: float = 9.81,
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
        self.da = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.0)))

    def _nu_forced(self, Tf: floatArrayLike, nu: floatArrayLike) -> floatArrayLike:
        """Nusselt number for forced convection."""
        rd = Air.relative_density(Tf, self.alt)
        Re = rd * np.abs(self.ws) * self.D / nu

        s = np.ones_like(Tf) * np.ones_like(nu) * np.ones_like(Re)
        z = s.shape == ()
        if z:
            s = np.array([1.0])

        B1 = 0.641 * s
        n = 0.471 * s

        # NB : (0.641/0.178)**(1/(0.633-0.471)) = 2721.4642715250125
        ix = np.logical_and(self.R <= 0.05, Re >= 2721.4642715250125)
        # NB : (0.641/0.048)**(1/(0.800-0.471)) = 2638.3210085195865
        jx = np.logical_and(self.R > 0.05, Re >= 2638.3210085195865)

        B1[ix] = 0.178
        B1[jx] = 0.048

        n[ix] = 0.633
        n[jx] = 0.800

        if z:
            B1 = B1[0]
            n = n[0]

        B2 = np.where(self.da < np.deg2rad(24.0), 0.68, 0.58)
        m1 = np.where(self.da < np.deg2rad(24.0), 1.08, 0.90)

        return np.maximum(0.42 + B2 * np.sin(self.da) ** m1, 0.55) * (B1 * Re**n)

    def _nu_natural(
        self,
        Tf: floatArrayLike,
        Td: floatArrayLike,
        nu: floatArrayLike,
    ) -> floatArrayLike:
        """Nusselt number for natural convection."""
        gr = self.D**3 * np.abs(Td) * self.g / ((Tf + 273.15) * nu**2)
        gp = gr * Air.prandtl(Tf)
        # gp[gp < 0.] = 0.
        ia = gp < 1.0e04
        A2, m2 = (np.zeros_like(Tf),) * 2
        A2[ia] = 0.850
        m2[ia] = 0.188
        A2[~ia] = 0.480
        m2[~ia] = 0.250
        return A2 * gp**m2

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
        Ta: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
        sigma: float = 5.67e-08,
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
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.

        Returns
        -------

        """
        super().__init__(
            Ta=Ta, D=D, epsilon=epsilon, sigma=sigma, zerok=273.0, **kwargs
        )
