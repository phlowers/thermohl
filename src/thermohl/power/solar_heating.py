# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List, Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun
from thermohl.power.power_term import PowerTerm


class _SRad:
    """Solar radiation calculator."""

    def __init__(self, clean: List[float], indus: List[float]):
        """Initialize the solar radiation calculator.

        Parameters
        ----------
        clean : List[float]
            Coefficients for polynomial function to compute atmospheric turbidity in clean air conditions.
        indus : List[float]
            Coefficients for polynomial function to compute atmospheric turbidity in industrial (polluted) air conditions.
        """
        self.clean = clean
        self.indus = indus

    def _solar_irradiance(
        self,
        lat: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation.

        Difference with IEEE version are neither turbidity or altitude influence.

        Parameters
        ----------
        lat : floatArrayLike
            Latitude in radians.
        month : intArrayLike
            Month (1-12).
        day : intArrayLike
            Day of the month.
        hour : floatArrayLike
            Hour of the day (0-24).

        Returns
        -------
        floatArrayLike
            Solar radiation value. Negative values are set to zero.
        """
        solar_altitude = sun.solar_altitude(lat, month, day, hour)
        atmospheric_coefficient = self.catm(np.rad2deg(solar_altitude))
        return np.where(solar_altitude > 0.0, atmospheric_coefficient, 0.0)

    def catm(
        self,
        solar_altitude_deg: floatArrayLike,
        turbidity_factor: Optional[floatArrayLike] = 0.0,
    ) -> floatArrayLike:
        """Compute coefficient for atmosphere turbidity.

        This method calculates the atmospheric turbidity coefficient using a polynomial
        function of the solar altitude. The coefficients of the polynomial are a weighted
        average of the clean air and industrial air coefficients, with the weights
        determined by the turbidity factor.

        Parameters
        ----------
        solar_altitude_deg : floatArrayLike
            Solar altitude in degrees.
        turbidity_factor : floatArrayLike
            Factor representing the atmospheric turbidity (0 for clean air, 1 for industrial air).

        Returns
        -------
        floatArrayLike
            Coefficient for atmospheric turbidity.
        """
        clean_air_factor = 1.0 - turbidity_factor
        A = clean_air_factor * self.clean[6] + turbidity_factor * self.indus[6]
        B = clean_air_factor * self.clean[5] + turbidity_factor * self.indus[5]
        C = clean_air_factor * self.clean[4] + turbidity_factor * self.indus[4]
        D = clean_air_factor * self.clean[3] + turbidity_factor * self.indus[3]
        E = clean_air_factor * self.clean[2] + turbidity_factor * self.indus[2]
        F = clean_air_factor * self.clean[1] + turbidity_factor * self.indus[1]
        G = clean_air_factor * self.clean[0] + turbidity_factor * self.indus[0]
        return (
            A * solar_altitude_deg**6
            + B * solar_altitude_deg**5
            + C * solar_altitude_deg**4
            + D * solar_altitude_deg**3
            + E * solar_altitude_deg**2
            + F * solar_altitude_deg**1
            + G
        )

    def __call__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        azimuth: floatArrayLike,
        turbidity_factor: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation.

        This method calculates the solar radiation based on geographical and temporal parameters.
        It computes the solar position (altitude and azimuth), the incidence angle,
        applies an altitude correction factor, and uses the atmospheric turbidity
        coefficient to determine the final solar radiation value.

        Parameters
        ----------
        latitude : floatArrayLike
            Latitude in radians.
        altitude : floatArrayLike
            Altitude in meters.
        azimuth : floatArrayLike
            Azimuth in radians.
        turbidity_factor : floatArrayLike
            Factor representing the atmospheric turbidity (0 for clean air, 1 for industrial air).
        month : intArrayLike
            Month (1-12).
        day : intArrayLike
            Day of the month.
        hour : floatArrayLike
            Hour of the day (0-24).

        Returns
        -------
        floatArrayLike
            Solar radiation value. Negative values are set to zero.
        """
        # Hs
        solar_altitude = sun.solar_altitude(latitude, month, day, hour)
        # Zs
        solar_azimuth = sun.solar_azimuth(latitude, month, day, hour)
        # theta
        incidence_angle = np.arccos(
            np.cos(solar_altitude) * np.cos(solar_azimuth - azimuth)
        )
        altitude_correction_factor = (
            1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude**2
        )
        atmospheric_turbidity_coefficient = self.catm(
            solar_altitude_deg=np.rad2deg(solar_altitude),
            turbidity_factor=turbidity_factor,
        )
        solar_radiation = (
            altitude_correction_factor
            * atmospheric_turbidity_coefficient
            * np.sin(incidence_angle)
        )
        return np.where(solar_radiation > 0.0, solar_radiation, 0.0)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
        self,
        latitude_deg: floatArrayLike,
        altitude_m: floatArrayLike,
        azimuth_deg: floatArrayLike,
        turbidity_factor: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        conductor_diameter_m: floatArrayLike,
        absorption_coefficient: floatArrayLike,
        solar_radiation_estimator: _SRad,
        precomputed_solar_radiation: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        """Initialize the solar heating term.

        This method initializes the solar heating term with geographical, temporal,
        and conductor parameters. It either computes the solar radiation using the
        provided estimator or uses a pre-computed value.

        Parameters
        ----------
        latitude_deg : floatArrayLike
            Latitude in degrees.
        altitude_m : floatArrayLike
            Altitude in meters.
        azimuth_deg : floatArrayLike
            Azimuth in degrees.
        turbidity_factor : floatArrayLike
            Turbidity factor (0 for clean air, 1 for industrial air).
        month : intArrayLike
            Month (1-12).
        day : intArrayLike
            Day of the month.
        hour : floatArrayLike
            Hour of the day (0-24).
        conductor_diameter_m : floatArrayLike
            Conductor diameter in meters.
        absorption_coefficient : floatArrayLike
            Absorption coefficient of the conductor surface.
        solar_radiation_estimator : _SRad
            Solar radiation estimator.
        precomputed_solar_radiation : Optional[floatArrayLike], optional
            Pre-computed solar radiation, by default None.
        **kwargs : Any
            Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.absorption_coefficient = absorption_coefficient
        if precomputed_solar_radiation is None:
            self.solar_radiation = solar_radiation_estimator(
                latitude=np.deg2rad(latitude_deg),
                altitude=altitude_m,
                azimuth=np.deg2rad(azimuth_deg),
                turbidity_factor=turbidity_factor,
                month=month,
                day=day,
                hour=hour,
            )
        else:
            self.solar_radiation = np.maximum(precomputed_solar_radiation, 0.0)
        self.conductor_diameter_m = conductor_diameter_m

    def value(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        r"""Compute solar heating.

        Parameters
        ----------
        conductor_temperature : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return (
            self.absorption_coefficient
            * self.solar_radiation
            * self.conductor_diameter_m
            * np.ones_like(conductor_temperature)
        )

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative with respect to conductor temperature.

        This method calculates the derivative of the solar heating with respect to
        the conductor temperature. Since the solar heating is independent of the
        conductor temperature, the derivative is always zero.

        Parameters
        ----------
        conductor_temperature : floatArrayLike
            Conductor temperature in degrees Celsius.

        Returns
        -------
        floatArrayLike
            Derivative of the solar heating with respect to conductor temperature.
            Always returns zero for all temperatures.
        """
        return np.zeros_like(conductor_temperature)
