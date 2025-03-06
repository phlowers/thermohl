# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Any, Callable

import numpy as np

from thermohl import floatArrayLike, floatArray, strListLike, intArray
from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_
from thermohl.solver.slv3t import Solver3T


class Solver3TL(Solver3T):

    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ):
        super().__init__(dic, joule, solar, convective, radiative, precipitation)
        self.update()

    def _morgan_coefficients(self) -> Tuple[floatArray, intArray]:
        """
        Calculate coefficients for heat flux between surface and core in steady state.

        Parameters:
        -----------
        D : float or numpy.ndarray
            The diameter of the core.
        d : float or numpy.ndarray
            The diameter of the surface.
        shape : Tuple[int, ...], optional
            The shape of the output arrays, default is (1,).

        Returns:
        --------
        Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[int]]
            - c : numpy.ndarray[float]
                Coefficient array for heat flux.
            - i : numpy.ndarray[int]
                Indices where surface diameter `d_` is greater than 0.
        """
        d = self.args.d * np.ones((self.args.max_len(),))
        i = np.nonzero(d > 0.0)[0]
        c = 1 / 13 * np.ones_like(d)
        c[i] = 1 / 21
        return c, i

    def average(self, ts, tc):
        """
        Compute average temperature given surface and core temperature.

        Unlike Solver3T, always use a regular mean even for non-homogeneous
        conductors.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        float or numpy.ndarray: Array of average temperatures.
        """
        return 0.5 * (ts + tc)

    def morgan(self, ts: floatArray, tc: floatArray) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        numpy.ndarray: Resulting array after applying the Morgan function.
        """
        c = self.mgc[0]
        return (tc - ts) - c * self.joule(ts, tc)

    def _steady_intensity_header(
        self, T: floatArrayLike, target: strListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.max_len()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.d, max_len)

        # pre-compute indexes
        js = np.nonzero(target_ == Solver_.Names.surf)[0]
        ja = np.nonzero(target_ == Solver_.Names.avg)[0]
        jc = np.nonzero(target_ == Solver_.Names.core)[0]

        def newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            ts = np.ones_like(tg) * np.nan
            tc = np.ones_like(tg) * np.nan

            ts[js] = Tmax[js]
            tc[js] = tg[js]

            ts[ja] = tg[ja]
            tc[ja] = 2 * Tmax[ja] - ts[ja]

            tc[jc] = Tmax[jc]
            ts[jc] = tg[jc]

            return ts, tc

        return Tmax, newtheader
