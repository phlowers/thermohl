# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun as sun
from thermohl.power import PowerTerm


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

        Args:
            lat (float | numpy.ndarray): Latitude.
            azm (float | numpy.ndarray): Azimuth.
            al (float | numpy.ndarray): Albedo.
            month (int | numpy.ndarray): Month number (must be between 1 and 12).
            day (int | numpy.ndarray): Day of the month (must be between 1 and 28, 29, 30 or 31 depending on month).
            hour (float | numpy.ndarray): Hour of the day (solar, must be between 0 and 23).
            D (float | numpy.ndarray): external diameter.
            alpha (float | numpy.ndarray): Solar absorption coefficient.
            srad (float | numpy.ndarray | None): Optional precomputed solar radiation term.
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

        Args:
            T (float | numpy.ndarray): Conductor temperature (°C).

        Returns:
            float | numpy.ndarray: Power term value (W·m⁻¹).

        """
        return self.alpha * self.srad * self.D * np.ones_like(T)

    def derivative(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        """Compute solar heating derivative."""
        return np.zeros_like(conductor_temperature)
