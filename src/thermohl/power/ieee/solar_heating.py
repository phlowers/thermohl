# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Any


from thermohl import floatArrayLike, intArrayLike
from thermohl.power import _SRad, SolarHeatingBase


class SolarHeating(SolarHeatingBase):
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
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude.
        alt : float or np.ndarray
            Altitude.
        azm : float or np.ndarray
            Azimuth.
        tb : float or np.ndarray
            Air pollution from 0 (clean) to 1 (polluted).
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
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        est = _SRad(
            [
                -4.22391e01,
                +6.38044e01,
                -1.9220e00,
                +3.46921e-02,
                -3.61118e-04,
                +1.94318e-06,
                -4.07608e-09,
            ],
            [
                +5.31821e01,
                +1.4211e01,
                +6.6138e-01,
                -3.1658e-02,
                +5.4654e-04,
                -4.3446e-06,
                +1.3236e-08,
            ],
        )
        super().__init__(
            lat, alt, azm, tb, month, day, hour, D, alpha, est, srad, **kwargs
        )
