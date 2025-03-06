# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power import PowerTerm
from thermohl.power.cigre import Air


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
        """
        Calculate the Nusselt number for forced convection.

        Parameters:
        -----------
        Tf : float or numpy.ndarray
            Array of fluid temperatures.
        nu : float or numpy.ndarray
            Array of kinematic viscosities.

        Returns:
        --------
        float or numpy.ndarray
            Array of Nusselt numbers for forced convection.
        Notes:
        ------
        The function calculates the Nusselt number based on the relative density of air,
        Reynolds number, and empirical correlations. The correlations are adjusted based
        on the Reynolds number and the ratio of characteristic length to diameter (R).
        The function also considers the angle of attack (da) to adjust the coefficients
        for the Nusselt number calculation.
        """
        relative_density = Air.relative_density(Tf, self.alt)
        Re = relative_density * np.abs(self.ws) * self.D / nu

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
        """
        Calculate the Nusselt number for natural convection.

        Parameters
        ----------
        Tf : float or numpy.ndarray
            Film temperature in degrees Celsius.
        Td : float or numpy.ndarray
            Temperature difference in degrees Celsius.
        nu : float or numpy.ndarray
            Kinematic viscosity in m^2/s.

        Returns
        -------
        float or numpy.ndarray
            Nusselt number for natural convection.

        Notes
        -----
        The function calculates the Grashof number (gr) and the product of the Grashof
        number and the Prandtl number (gp). It then uses these values to determine the
        Nusselt number based on empirical correlations for different ranges of gp.

        """
        gr = self.D**3 * np.abs(Td) * self.g / ((Tf + 273.15) * nu**2)
        gp = gr * Air.prandtl(Tf)
        ia = gp < 1.0e04
        A2 = np.ones_like(gp) * 0.480
        m2 = np.ones_like(gp) * 0.250

        if len(gp.shape) == 0:
            if ia:
                A2 = 0.850
                m2 = 0.188
        else:
            A2[ia] = 0.850
            m2[ia] = 0.188
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
