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
from thermohl.power.convective_cooling import compute_wind_attack_angle
from thermohl.power.cigre import Air


class ConvectiveCooling(PowerTerm):
    """Convective cooling term."""

    def __init__(
        self,
        altitude: floatArrayLike,
        cable_azimuth: floatArrayLike,
        ambient_temperature: floatArrayLike,
        wind_speed: floatArrayLike,
        wind_azimuth: floatArrayLike,
        outer_diameter: floatArrayLike,
        roughness_ratio: floatArrayLike,
        g: float = 9.81,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Args:
            altitude (float | numpy.ndarray): Altitude (m).
            cable_azimuth (float | numpy.ndarray): Azimuth (deg).
            ambient_temperature (float | numpy.ndarray): Ambient temperature (°C).
            wind_speed (float | numpy.ndarray): Wind speed (m·s⁻¹).
            wind_azimuth (float | numpy.ndarray): wind azimuth regarding north (deg).
            outer_diameter (float | numpy.ndarray): External diameter (m).
            roughness_ratio (float | numpy.ndarray): Cable roughness (—).
            g (float, optional): Gravitational acceleration (m·s⁻²). The default is 9.81.

        """
        self.altitude = altitude
        self.ambient_temp = ambient_temperature
        self.wind_speed = wind_speed
        self.outer_diameter = outer_diameter
        self.roughness_ratio = roughness_ratio
        self.gravity = g
        self.wind_attack_angle = compute_wind_attack_angle(cable_azimuth, wind_azimuth)

    def _nu_forced(
        self, film_temperature: floatArrayLike, kinematic_viscosity: floatArrayLike
    ) -> floatArrayLike:
        """
        Calculate the Nusselt number for forced convection.

        Args:
            film_temperature (float | numpy.ndarray): Film temperature (°C).
            kinematic_viscosity (float | numpy.ndarray): Kinematic viscosity (m²·s⁻¹).

        Returns:
            float | numpy.ndarray: Nusselt number for forced convection.

        Notes:
            The function calculates the Nusselt number based on the relative density of air,
            the Reynolds number, and empirical correlations. The correlations are adjusted
            depending on the Reynolds number and the roughness ratio roughness_ratio. The function also
            considers the angle of attack (da) to adjust the coefficients.
        """
        relative_density = Air.relative_density(film_temperature, self.altitude)
        reynolds = (
            relative_density
            * np.abs(self.wind_speed)
            * self.outer_diameter
            / kinematic_viscosity
        )

        s = (
            np.ones_like(film_temperature)
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

        B2 = np.where(self.wind_attack_angle < np.deg2rad(24.0), 0.68, 0.58)
        m1 = np.where(self.wind_attack_angle < np.deg2rad(24.0), 1.08, 0.90)

        return np.maximum(0.42 + B2 * np.sin(self.wind_attack_angle) ** m1, 0.55) * (
            B1 * reynolds**n
        )

    def _nu_natural(
        self,
        film_temperature: floatArrayLike,
        temperature_delta: floatArrayLike,
        kinematic_viscosity: floatArrayLike,
    ) -> floatArrayLike:
        """
        Calculate the Nusselt number for natural convection.

        Args:
            film_temperature (float | numpy.ndarray): Film temperature (°C).
            temperature_delta (float | numpy.ndarray): Temperature difference (°C).
            kinematic_viscosity (float | numpy.ndarray): Kinematic viscosity (m²·s⁻¹).

        Returns:
            float | numpy.ndarray: Nusselt number for natural convection.

        Notes:
            The function calculates the Grashof number (gr) and the product of the Grashof
            number and the Prandtl number (gp). It then uses these values to determine the
            Nusselt number based on empirical correlations for different ranges of gp.

        """
        grashof = (
            self.outer_diameter**3
            * np.abs(temperature_delta)
            * self.gravity
            / ((film_temperature + 273.15) * kinematic_viscosity**2)
        )
        gr_prandtl = grashof * Air.prandtl(film_temperature)
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

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Args:
            conductor_temperature (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        film_temperature = 0.5 * (conductor_temperature + self.ambient_temp)
        temperature_delta = conductor_temperature - self.ambient_temp
        kinematic_viscosity = Air.kinematic_viscosity(film_temperature)
        # nu[nu < 1.0E-06] = 1.0E-06
        thermal_conductivity = Air.thermal_conductivity(film_temperature)
        # lm[lm < 0.01] = 0.01
        nusselt_forced = self._nu_forced(film_temperature, kinematic_viscosity)
        nusselt_natural = self._nu_natural(
            film_temperature, temperature_delta, kinematic_viscosity
        )
        return (
            np.pi
            * thermal_conductivity
            * (conductor_temperature - self.ambient_temp)
            * np.maximum(nusselt_forced, nusselt_natural)
        )
