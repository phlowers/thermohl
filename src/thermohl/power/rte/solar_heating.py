# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import Optional, Any

import numpy as np

from thermohl import floatArrayLike, intArrayLike, sun
from thermohl.power import SolarHeatingBase, _SRad

CLEAN_AIR_COEFFICIENTS = [
    -42.0,
    +63.8,
    -1.922,
    0.03469,
    -3.61e-04,
    +1.943e-06,
    -4.08e-09,
]
POLLUTED_AIR_COEFFICIENTS = [0, 0, 0, 0, 0, 0, 0]

solar_radiation = _SRad(clean=CLEAN_AIR_COEFFICIENTS, indus=POLLUTED_AIR_COEFFICIENTS)


def solar_irradiance(
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
    atmospheric_coefficient = solar_radiation.catm(np.rad2deg(solar_altitude))
    return np.where(solar_altitude > 0.0, atmospheric_coefficient, 0.0)


class SolarHeating(SolarHeatingBase):
    def __init__(
        self,
        lat: floatArrayLike,
        azm: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        D: floatArrayLike,
        alpha: floatArrayLike,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Build with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
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
        srad : xxx
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        for k in ["alt", "tb"]:
            if k in kwargs.keys():
                kwargs.pop(k)
        super().__init__(
            lat=lat,
            alt=0.0,
            azm=azm,
            tb=0.0,
            month=month,
            day=day,
            hour=hour,
            D=D,
            alpha=alpha,
            est=solar_radiation,
            srad=srad,
            **kwargs,
        )
