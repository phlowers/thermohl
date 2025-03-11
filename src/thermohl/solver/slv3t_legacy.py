# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Dict, Any

import numpy as np
import pandas as pd

from thermohl import floatArrayLike, floatArray, strListLike, intArray
from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_, _DEFPARAM as DP, _set_dates, reshape
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T
from thermohl.utils import quasi_newton_2d


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

    def joule(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        """
        Calculate the Joule heating effect.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        float or numpy.ndarray: The calculated Joule heating values.

        Notes:
        - The function computes the average temperature `temperature` as the mean of `ts` and `tc`.
        - There is no adjustment for bimaterial cables as in the non-legacy version.
        - Finally, it returns the Joule heating values based on the adjusted temperatures.
        """
        return self.jh.value(0.5 * (ts + tc))

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

    def steady_temperature(
        self,
        Tsg: Optional[floatArrayLike] = None,
        Tcg: Optional[floatArrayLike] = None,
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the steady-state temperature distribution.
        Parameters:
        -----------
        Tsg : Optional[float or numpy.ndarray], default=None
            Initial guess for the surface temperature. If None, ambient temperature is used.
        Tcg : Optional[float or numpy.ndarray], default=None
            Initial guess for the core temperature. If None, 1.5 times the absolute value of ambient temperature is used.
        tol : float, default=DP.tol
            Tolerance for the quasi-Newton solver.
        maxiter : int, default=DP.maxiter
            Maximum number of iterations for the quasi-Newton solver.
        return_err : bool, default=False
            If True, the error of the solution is included in the returned DataFrame.
        return_power : bool, default=True
            If True, power-related values are included in the returned DataFrame.
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the steady-state temperatures and optionally the error and power-related values.
        """

        # if no guess provided, use ambient temp
        shape = (self.args.max_len(),)
        Tsg = Tsg if Tsg is not None else 1.0 * self.args.Ta
        Tcg = Tcg if Tcg is not None else 1.5 * np.abs(self.args.Ta)
        Tsg_ = Tsg * np.ones(shape)
        Tcg_ = Tcg * np.ones(shape)

        # solve system
        x, y, cnt, err = quasi_newton_2d(
            f1=self.balance,
            f2=self.morgan,
            x0=Tsg_,
            y0=Tcg_,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or cnt == maxiter:
            print(f"rstat_analytic max err is {np.max(err):.3E} in {cnt:d} iterations")

        # format output
        df = pd.DataFrame(
            {
                Solver_.Names.tsurf: x,
                Solver_.Names.tavg: 0.5 * (x + y),
                Solver_.Names.tcore: y,
            }
        )

        if return_err:
            df[Solver_.Names.err] = err

        if return_power:
            df[Solver_.Names.pjle] = self.joule(x, y)
            df[Solver_.Names.psol] = self.sh.value(x)
            df[Solver_.Names.pcnv] = self.cc.value(x)
            df[Solver_.Names.prad] = self.rc.value(x)
            df[Solver_.Names.ppre] = self.pc.value(x)

        return df

    def steady_intensity(
        self,
        T: floatArrayLike = np.array([]),
        target: strListLike = "auto",
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_temp: bool = True,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the steady-state intensity for a given temperature profile.
        Parameters:
        -----------
        T : float or numpy.ndarray, optional
            Initial temperature profile. Default is an empty numpy array.
        target : str or List[str], optional
            Target specification for the solver. Default is "auto".
        tol : float, optional
            Tolerance for the solver. Default is DP.tol.
        maxiter : int, optional
            Maximum number of iterations for the solver. Default is DP.maxiter.
        return_err : bool, optional
            If True, return the error in the output DataFrame. Default is False.
        return_temp : bool, optional
            If True, return the temperature profiles in the output DataFrame. Default is True.
        return_power : bool, optional
            If True, return the power profiles in the output DataFrame. Default is True.
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the steady-state intensity and optionally the error, temperature profiles, and power profiles.
        """

        max_len = self.args.max_len()
        Tmax_ = T * np.ones(max_len)

        target_ = self._check_target(target, self.args.d, max_len)

        # pre-compute indexes
        js = np.nonzero(target_ == Solver_.Names.surf)[0]
        ja = np.nonzero(target_ == Solver_.Names.avg)[0]
        jc = np.nonzero(target_ == Solver_.Names.core)[0]

        def _newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            ts = np.ones_like(tg) * np.nan
            tc = np.ones_like(tg) * np.nan

            ts[js] = Tmax_[js]
            tc[js] = tg[js]

            ts[ja] = tg[ja]
            tc[ja] = 2 * Tmax_[ja] - ts[ja]

            tc[jc] = Tmax_[jc]
            ts[jc] = tg[jc]

            return ts, tc

        def balance(i: floatArray, tg: floatArray) -> floatArrayLike:
            ts, tc = _newtheader(i, tg)
            return self.balance(ts, tc)

        def morgan(i: floatArray, tg: floatArray) -> floatArray:
            ts, tc = _newtheader(i, tg)
            return self.morgan(ts, tc)

        # solve system
        s = Solver1T(
            self.args.__dict__,
            type(self.jh),
            type(self.sh),
            type(self.cc),
            type(self.rc),
            type(self.pc),
        )
        r = s.steady_intensity(Tmax_, tol=1.0, maxiter=8, return_power=False)
        x, y, cnt, err = quasi_newton_2d(
            balance,
            morgan,
            r[Solver_.Names.transit].values,
            Tmax_,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or cnt == maxiter:
            print(f"rstat_analytic max err is {np.max(err):.3E} in {cnt:d} iterations")

        # format output
        df = pd.DataFrame({Solver_.Names.transit: x})

        if return_err:
            df["err"] = err

        if return_temp or return_power:
            ts, tc = _newtheader(x, y)
            ta = 0.5 * (ts + tc)

            if return_temp:
                df[Solver_.Names.tsurf] = ts
                df[Solver_.Names.tavg] = ta
                df[Solver_.Names.tcore] = tc

            if return_power:
                df[Solver_.Names.pjle] = self.jh.value(ta)
                df[Solver_.Names.psol] = self.sh.value(ts)
                df[Solver_.Names.pcnv] = self.cc.value(ts)
                df[Solver_.Names.prad] = self.rc.value(ts)
                df[Solver_.Names.ppre] = self.pc.value(ts)

        return df
