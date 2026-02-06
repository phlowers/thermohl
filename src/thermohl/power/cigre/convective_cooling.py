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

        Args:
            alt (float | numpy.ndarray): Altitude (m).
            azm (float | numpy.ndarray): Azimuth (deg).
            Ta (float | numpy.ndarray): Ambient temperature (°C).
            ws (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wa (float | numpy.ndarray): Wind angle regarding north (deg).
            D (float | numpy.ndarray): External diameter (m).
            R (float | numpy.ndarray): Cable roughness (—).
            g (float, optional): Gravitational acceleration (m·s⁻²). The default is 9.81.

        """
        self.altitude_m = alt
        self.ambient_temp_c = Ta
        self.wind_speed_ms = ws
        self.outer_diameter_m = D
        self.roughness_ratio = R
        self.gravity_ms2 = g
        self.attack_angle_rad = np.arcsin(np.sin(np.deg2rad(np.abs(azm - wa) % 180.0)))

    def _nu_forced(
        self, film_temp_c: floatArrayLike, kinematic_viscosity: floatArrayLike
    ) -> floatArrayLike:
        """
        Calculate the Nusselt number for forced convection.

        Args:
            film_temp_c (float | numpy.ndarray): Film temperature (°C).
            kinematic_viscosity (float | numpy.ndarray): Kinematic viscosity (m²·s⁻¹).

        Returns:
            float | numpy.ndarray: Nusselt number for forced convection.

        Notes:
            The function calculates the Nusselt number based on the relative density of air,
            the Reynolds number, and empirical correlations. The correlations are adjusted
            depending on the Reynolds number and the roughness ratio R. The function also
            considers the angle of attack (da) to adjust the coefficients.
        """
        relative_density = Air.relative_density(film_temp_c, self.altitude_m)
        reynolds = (
            relative_density
            * np.abs(self.wind_speed_ms)
            * self.outer_diameter_m
            / kinematic_viscosity
        )

        s = (
            np.ones_like(film_temp_c)
            * np.ones_like(kinematic_viscosity)
            * np.ones_like(reynolds)
        )
        z = s.shape == ()
        if z:
            s = np.array([1.0])

        B1 = 0.641 * s
        n = 0.471 * s

        # NB : (0.641/0.178)**(1/(0.633-0.471)) = 2721.4642715250125
        ix = np.logical_and(
            self.roughness_ratio <= 0.05, reynolds >= 2721.4642715250125
        )
        # NB : (0.641/0.048)**(1/(0.800-0.471)) = 2638.3210085195865
        jx = np.logical_and(self.roughness_ratio > 0.05, reynolds >= 2638.3210085195865)

        B1[ix] = 0.178
        B1[jx] = 0.048

        n[ix] = 0.633
        n[jx] = 0.800

        if z:
            B1 = B1[0]
            n = n[0]

        B2 = np.where(self.attack_angle_rad < np.deg2rad(24.0), 0.68, 0.58)
        m1 = np.where(self.attack_angle_rad < np.deg2rad(24.0), 1.08, 0.90)

        return np.maximum(0.42 + B2 * np.sin(self.attack_angle_rad) ** m1, 0.55) * (
            B1 * reynolds**n
        )

    def _nu_natural(
        self,
        film_temp_c: floatArrayLike,
        temp_delta_c: floatArrayLike,
        kinematic_viscosity: floatArrayLike,
    ) -> floatArrayLike:
        """
        Calculate the Nusselt number for natural convection.

        Args:
            film_temp_c (float | numpy.ndarray): Film temperature (°C).
            temp_delta_c (float | numpy.ndarray): Temperature difference (°C).
            kinematic_viscosity (float | numpy.ndarray): Kinematic viscosity (m²·s⁻¹).

        Returns:
            float | numpy.ndarray: Nusselt number for natural convection.

        Notes:
            The function calculates the Grashof number (gr) and the product of the Grashof
            number and the Prandtl number (gp). It then uses these values to determine the
            Nusselt number based on empirical correlations for different ranges of gp.

        """
        grashof = (
            self.outer_diameter_m**3
            * np.abs(temp_delta_c)
            * self.gravity_ms2
            / ((film_temp_c + 273.15) * kinematic_viscosity**2)
        )
        gr_prandtl = grashof * Air.prandtl(film_temp_c)
        ia = gr_prandtl < 1.0e04
        A2 = np.ones_like(gr_prandtl) * 0.480
        m2 = np.ones_like(gr_prandtl) * 0.250

        if len(gr_prandtl.shape) == 0:
            if ia:
                A2 = 0.850
                m2 = 0.188
        else:
            A2[ia] = 0.850
            m2[ia] = 0.188
        return A2 * gr_prandtl**m2

    def value(self, conductor_temp_c: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Args:
            conductor_temp_c (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        film_temp_c = 0.5 * (conductor_temp_c + self.ambient_temp_c)
        temp_delta_c = conductor_temp_c - self.ambient_temp_c
        kinematic_viscosity = Air.kinematic_viscosity(film_temp_c)
        # nu[nu < 1.0E-06] = 1.0E-06
        thermal_conductivity = Air.thermal_conductivity(film_temp_c)
        # lm[lm < 0.01] = 0.01
        nusselt_forced = self._nu_forced(film_temp_c, kinematic_viscosity)
        nusselt_natural = self._nu_natural(
            film_temp_c, temp_delta_c, kinematic_viscosity
        )
        return (
            np.pi
            * thermal_conductivity
            * (conductor_temp_c - self.ambient_temp_c)
            * np.maximum(nusselt_forced, nusselt_natural)
        )
