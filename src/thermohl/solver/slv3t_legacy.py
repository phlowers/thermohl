# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Any, Callable, Final, Dict

import numpy as np

from thermohl import floatArrayLike, floatArray, strListLike, intArray
from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_, Args
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

        Returns:
        --------
        Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[int]]
            - heat_flux_coefficients : numpy.ndarray[float]
                Coefficient array for heat flux.
            - indices_non_zero_diameter : numpy.ndarray[int]
                Indices where core diameter is greater than 0.
                When conductors are uniform, core diameter is equal to 0.0.
                When conductors are bimetallic, core diameter is greater than 0.0.
        """
        UNIFORM_CONDUCTOR_COEFFICIENT: Final[float] = 1 / 13
        BIMETALLIC_CONDUCTOR_COEFFICIENT: Final[float] = 1 / 21

        core_diameter_array = self.args.d * np.ones((self.args.max_len(),))
        indices_non_zero_diameter = np.nonzero(core_diameter_array > 0.0)[0]
        heat_flux_coefficients = UNIFORM_CONDUCTOR_COEFFICIENT * np.ones_like(
            core_diameter_array
        )
        heat_flux_coefficients[indices_non_zero_diameter] = (
            BIMETALLIC_CONDUCTOR_COEFFICIENT
        )
        return heat_flux_coefficients, indices_non_zero_diameter

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
        morgan_coefficient = self.mgc[0]
        return (tc - ts) - morgan_coefficient * self.joule(ts, tc)

    def _steady_intensity_header(
        self, T: floatArrayLike, target: strListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.max_len()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.d, max_len)

        # pre-compute indexes
        surface_indices = np.nonzero(target_ == Solver_.Names.surf)[0]
        average_indices = np.nonzero(target_ == Solver_.Names.avg)[0]
        core_indices = np.nonzero(target_ == Solver_.Names.core)[0]

        def newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            ts = np.ones_like(tg) * np.nan
            tc = np.ones_like(tg) * np.nan

            ts[surface_indices] = Tmax[surface_indices]
            tc[surface_indices] = tg[surface_indices]

            ts[average_indices] = tg[average_indices]
            tc[average_indices] = 2 * Tmax[average_indices] - ts[average_indices]

            tc[core_indices] = Tmax[core_indices]
            ts[core_indices] = tg[core_indices]

            return ts, tc

        return Tmax, newtheader

    def _morgan_transient(self):
        """Morgan coefficients for transient temperature."""
        c1, _ = self.mgc
        c2 = 0.5 * np.ones_like(c1)
        return c1, c2

    def transient_temperature_legacy(
        self,
        time: floatArray = np.array([]),
        Ts0: Optional[floatArrayLike] = None,
        Tc0: Optional[floatArrayLike] = None,
        tau: float = 600.0,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature with legacy method.

        Parameters
        ----------
        time : numpy.ndarray
            A 1D array with times (in seconds) when the temperature needs to be
            computed. The array must contain increasing values (undefined
            behaviour otherwise).
        Ts0 : float
            Initial surface temperature. If set to None, the ambient temperature from
            internal dict will be used. The default is None.
        Tc0 : float
            Initial core temperature. If set to None, the ambient temperature from
            internal dict will be used. The default is None.
        tau: float
            A time-constant to add some inertia. The default is 600.
        return_power : bool, optional
            Return power term values. The default is False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with temperature and other results (depending on inputs)
            in the keys.

        """

        # get sizes (n for input dict entries, N for time)
        n = self.args.max_len()
        N = len(time)
        if N < 2:
            raise ValueError()

        # get initial temperature
        Ts0 = Ts0 if Ts0 is not None else self.args.Ta
        Tc0 = Tc0 if Tc0 is not None else 1.0 + Ts0

        # shortcuts for time-loop
        imc = 1.0 / (self.args.m * self.args.c)

        # init
        ts = np.zeros((N, n))
        ta = np.zeros((N, n))
        tc = np.zeros((N, n))
        dT = np.zeros((N, n))

        ts[0, :] = Ts0
        tc[0, :] = Tc0
        ta[0, :] = self.average(ts[0, :], tc[0, :])
        dT[0, :] = tc[0, :] - ts[0, :]

        for i in range(1, len(time)):
            bal = self.balance(ts[i - 1, :], tc[i - 1, :])
            dti = time[i] - time[i - 1]
            ta[i, :] = ta[i - 1, :] + dti * imc * bal
            dT[i, :] = (1.0 - dti / tau) * dT[i - 1, :] + (
                dti / tau * self.mgc[0] * self.jh.value(ta[i, :])
            )
            tc[i, :] = ta[i, :] + 0.5 * dT[i, :]
            ts[i, :] = ta[i, :] - 0.5 * dT[i, :]

        return self._transient_temperature_results(time, ts, ta, tc, return_power, n)
