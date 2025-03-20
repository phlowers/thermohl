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


class SolarRadiation:

    def __init__(self, clean: List[float], indus: List[float]):
        self.clean = clean
        self.indus = indus

    def atmosphere_coefficients(self, solar_altitude: floatArrayLike, turbidity: floatArrayLike) -> floatArrayLike:
        """Compute coefficient for atmosphere turbidity."""
        A = (1.0 - turbidity) * self.clean[6] + turbidity * self.indus[6]
        B = (1.0 - turbidity) * self.clean[5] + turbidity * self.indus[5]
        C = (1.0 - turbidity) * self.clean[4] + turbidity * self.indus[4]
        D = (1.0 - turbidity) * self.clean[3] + turbidity * self.indus[3]
        E = (1.0 - turbidity) * self.clean[2] + turbidity * self.indus[2]
        F = (1.0 - turbidity) * self.clean[1] + turbidity * self.indus[1]
        G = (1.0 - turbidity) * self.clean[0] + turbidity * self.indus[0]
        return A * solar_altitude**6 + B * solar_altitude**5 + C * solar_altitude**4 + D * solar_altitude**3 + E * solar_altitude**2 + F * solar_altitude**1 + G

    def __call__(
        self,
        latitude: floatArrayLike,
        altitude: floatArrayLike,
        line_azimut: floatArrayLike,
        trb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
    ) -> floatArrayLike:
        """Compute solar radiation."""
        solar_altitude = sun.solar_altitude(latitude, month, day, hour)
        solar_azimut = sun.solar_azimuth(latitude, month, day, hour)
        theta = np.arccos(np.cos(solar_altitude) * np.cos(solar_azimut - line_azimut))
        K = 1.0 + 1.148e-04 * altitude - 1.108e-08 * altitude ** 2
        Q = self.atmosphere_coefficients(np.rad2deg(solar_altitude), trb)
        solar_radiation = K * Q * np.sin(theta)
        return np.where(solar_radiation > 0.0, solar_radiation, 0.0)


class SolarHeatingBase(PowerTerm):
    """Solar heating term."""

    def __init__(
        self,
        lat: floatArrayLike,
        alt: floatArrayLike,
        azm: floatArrayLike,
        tb: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        D: floatArrayLike,
        alpha: floatArrayLike,
        est: SolarRadiation,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        self.alpha = alpha
        if srad is None:
            self.srad = est(np.deg2rad(lat), alt, np.deg2rad(azm), tb, month, day, hour)
        else:
            self.srad = np.maximum(srad, 0.0)
        self.D = D

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute solar heating.

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

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(conductor_temperature)
