# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd

from thermohl import floatArrayLike, floatArray, strListLike, intArray
from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_, _DEFPARAM as DP, _set_dates, reshape
from thermohl.solver.slv1t import Solver1T
from thermohl.utils import quasi_newton_2d


def _profile_mom(ts: float, tc: float, r: floatArrayLike, re: float) -> floatArrayLike:
    """Analytic temperature profile for steady heat equation in cylinder (mono-mat)."""
    return ts + (tc - ts) * (1.0 - (r / re) ** 2)


def _phi(r: floatArrayLike, ri: floatArrayLike, re: floatArrayLike) -> floatArrayLike:
    """Primitive function used in _profile_bim*** functions."""
    ri2 = ri**2
    return (0.5 * (r**2 - ri2) - ri2 * np.log(r / ri)) / (re**2 - ri2)


def _profile_bim_avg_coeffs(
    ri: floatArrayLike, re: floatArrayLike
) -> tuple[floatArrayLike, floatArrayLike]:
    ri2 = ri**2
    re2 = re**2
    a = 0.5 * (re2 - ri2) ** 2 - re2 * ri2 * (2.0 * np.log(re / ri) - 1.0) - ri**4
    b = 2.0 * re2 * (re2 - ri2) * _phi(re, ri, re)
    return a, b


def _profile_bim_avg(
    ts: floatArrayLike, tc: floatArrayLike, ri: floatArrayLike, re: floatArrayLike
) -> floatArrayLike:
    """Analytical formulation for average temperature in _profile_bim."""
    a, b = _profile_bim_avg_coeffs(ri, re)
    return tc - (a / b) * (tc - ts)


class Solver3T(Solver_):

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

    def _morgan_coefficients(
        self,
    ) -> Tuple[floatArray, floatArray, floatArray, intArray]:
        """
        Calculate coefficients for heat flux between surface and core in steady state.

        Returns:
        --------
        Tuple[numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[float], numpy.ndarray[int]]
            - c : numpy.ndarray[float]
                Coefficient array for heat flux.
            - D_ : numpy.ndarray[float]
                Array of core diameters, broadcasted to the shape of `c`.
            - d_ : numpy.ndarray[float]
                Array of surface diameters, broadcasted to the shape of `c`.
            - i : numpy.ndarray[int]
                Indices where surface diameter `d_` is greater than 0.
        """
        c = 0.5 * np.ones((self.args.max_len(),))
        D = self.args.D * np.ones_like(c)
        d = self.args.d * np.ones_like(c)
        i = np.nonzero(d > 0.0)[0]
        c[i] -= (d[i] ** 2 / (D[i] ** 2 - d[i] ** 2)) * np.log(D[i] / d[i])
        return c, D, d, i

    def update(self) -> None:
        """
        Updates the solver's internal state by reinitializing several components
        and recalculating the Morgan coefficients.
        This method performs the following steps:
        1. Extends the arguments to their maximum length.
        2. Reinitializes the `jh`, `sh`, `cc`, `rc`, and `pc` components using the updated arguments.
        3. Recalculates the Morgan coefficients using the updated dimensions.
        4. Compresses the arguments.
        Returns:
            None
        """
        self.args.extend_to_max_len()
        self.jh.__init__(**self.args.__dict__)
        self.sh.__init__(**self.args.__dict__)
        self.cc.__init__(**self.args.__dict__)
        self.rc.__init__(**self.args.__dict__)
        self.pc.__init__(**self.args.__dict__)

        self.mgc = self._morgan_coefficients()

        self.args.compress()

    def average(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        """
        Compute average temperature given surface and core temperature.

        This formula is based on analytical solution in steady-state mode. For
        single material, the formula reduces itself to an usual mean; for
        bi-material conductors, we have geometrical terms to take into account.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        float or numpy.ndarray: Array of average temperatures.
        """
        ta = 0.5 * (ts + tc)
        _, D, d, ix = self.mgc
        ta[ix] = _profile_bim_avg(ts[ix], tc[ix], 0.5 * d[ix], 0.5 * D[ix])
        return ta

    def joule(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        """
        Calculate the Joule heating effect.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        float or numpy.ndarray: The calculated Joule heating values.

        Notes:
        - The function computes the average temperature `temperature`.
        - Returns the Joule heating values based on the adjusted temperatures.
        """
        ta = self.average(ts, tc)
        return self.jh.value(ta)

    def balance(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        """
        Calculate the thermal balance.

        This method computes the thermal balance by summing the joule heating,
        specific heat, and subtracting the contributions from the cooling
        components (convection, radiation, and conduction).

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        float or numpy.ndarray: The resulting thermal balance.
        """
        return (
            self.joule(ts, tc)
            + self.sh.value(ts)
            - self.cc.value(ts)
            - self.rc.value(ts)
            - self.pc.value(ts)
        )

    def morgan(self, ts: floatArray, tc: floatArray) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Parameters:
        ts (numpy.ndarray): Array of surface temperatures.
        tc (numpy.ndarray): Array of core temperatures.

        Returns:
        numpy.ndarray: Resulting array after applying the Morgan function.
        """
        c, _, _, _ = self.mgc
        return (tc - ts) - c * self.joule(ts, tc) / (2.0 * np.pi * self.args.l)

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
        z = self.average(x, y)
        df = pd.DataFrame(
            {Solver_.Names.tsurf: x, Solver_.Names.tavg: z, Solver_.Names.tcore: y}
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

    def _morgan_transient(self):
        """Morgan coefficients for transient temperature."""
        c, D, d, ix = self.mgc
        c1 = c / (2.0 * np.pi * self.args.l)
        c2 = 0.5 * np.ones_like(c1)
        a, b = _profile_bim_avg_coeffs(0.5 * d[ix], 0.5 * D[ix])
        c2[ix] = a / b
        return c1, c2

    def _transient_temperature_results(self, time, ts, ta, tc, return_power, n):
        dr = {
            Solver_.Names.time: time,
            Solver_.Names.tsurf: ts,
            Solver_.Names.tavg: ta,
            Solver_.Names.tcore: tc,
        }

        if return_power:
            for power in Solver_.Names.powers():
                dr[power] = np.zeros_like(ts)

            for i in range(len(time)):
                dr[Solver_.Names.pjle][i, :] = self.joule(ts[i, :], tc[i, :])
                dr[Solver_.Names.psol][i, :] = self.sh.value(ts[i, :])
                dr[Solver_.Names.pcnv][i, :] = self.cc.value(ts[i, :])
                dr[Solver_.Names.prad][i, :] = self.rc.value(ts[i, :])
                dr[Solver_.Names.ppre][i, :] = self.pc.value(ts[i, :])

        if n == 1:
            keys = list(dr.keys())
            keys.remove(Solver_.Names.time)
            for k in keys:
                dr[k] = dr[k][:, 0]

        return dr

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        Ts0: Optional[floatArrayLike] = None,
        Tc0: Optional[floatArrayLike] = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

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

        # get month, day and hours
        month, day, hour = _set_dates(
            self.args.month, self.args.day, self.args.hour, time, n
        )

        # Two dicts, one (dc) with static quantities (with all elements of size
        # n), the other (de) with time-changing quantities (with all elements of
        # size N*n); uk is a list of keys that are in dc but not in de.
        de = dict(
            month=month,
            day=day,
            hour=hour,
            I=reshape(self.args.I, N, n),
            Ta=reshape(self.args.Ta, N, n),
            wa=reshape(self.args.wa, N, n),
            ws=reshape(self.args.ws, N, n),
            Pa=reshape(self.args.Pa, N, n),
            rh=reshape(self.args.rh, N, n),
            pr=reshape(self.args.pr, N, n),
        )
        del (month, day, hour)

        # shortcuts for time-loop
        c1, c2 = self._morgan_transient()
        imc = 1.0 / (self.args.m * self.args.c)

        # init
        ts = np.zeros((N, n))
        ta = np.zeros((N, n))
        tc = np.zeros((N, n))
        ts[0, :] = Ts0
        tc[0, :] = Tc0
        ta[0, :] = self.average(ts[0, :], tc[0, :])

        # main time loop
        for i in range(1, len(time)):
            for k in de.keys():
                self.args[k] = de[k][i, :]
            self.update()
            bal = self.balance(ts[i - 1, :], tc[i - 1, :])
            ta[i, :] = ta[i - 1, :] + (time[i] - time[i - 1]) * bal * imc
            mrg = c1 * (self.jh.value(ta[i, :]) - bal)
            tc[i, :] = ta[i, :] + c2 * mrg
            ts[i, :] = tc[i, :] - mrg

        return self._transient_temperature_results(time, ts, ta, tc, return_power, n)

    @staticmethod
    def _check_target(target, d, max_len):
        """
        Validates and processes the target temperature input.

        Parameters:
        target (str or list): The target temperature(s) to be validated. It can be:
            - "auto": which sets the target to None.
            - A string: which should be one of Solver_.Names.surf, Solver_.Names.avg, or Solver_.Names.core.
            - A list of strings: where each string should be one of Solver_.Names.surf, Solver_.Names.avg, or Solver_.Names.core.
        max_len (int): The expected length of the target list if target is a list.

        Returns:
        numpy.ndarray: An array of target temperatures if the input is valid.

        Raises:
        ValueError: If the target is a string not in the allowed values, or if the target list length does not match max_len, or if any element in the target list is not in the allowed values.
        """
        # check target
        if target == "auto":
            d_ = d * np.ones(max_len)
            target_ = np.array(
                [
                    Solver_.Names.core if d_[i] > 0.0 else Solver_.Names.avg
                    for i in range(max_len)
                ]
            )
        elif isinstance(target, str):
            if target not in [
                Solver_.Names.surf,
                Solver_.Names.avg,
                Solver_.Names.core,
            ]:
                raise ValueError(
                    f"Target temperature should be in "
                    f"{[Solver_.Names.surf, Solver_.Names.avg, Solver_.Names.core]};"
                    f" got {target} instead."
                )
            else:
                target_ = np.array([target for _ in range(max_len)])
        else:
            if len(target) != max_len:
                raise ValueError()
            for t in target:
                if t not in [
                    Solver_.Names.surf,
                    Solver_.Names.avg,
                    Solver_.Names.core,
                ]:
                    raise ValueError()
            target_ = np.array(target)
        return target_

    def _steady_intensity_header(
        self, T: floatArrayLike, target: strListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.max_len()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.d, max_len)

        # pre-compute indexes
        c, D, d, ix = self.mgc
        a, b = _profile_bim_avg_coeffs(0.5 * d, 0.5 * D)

        js = np.nonzero(target_ == Solver_.Names.surf)[0]
        ja = np.nonzero(target_ == Solver_.Names.avg)[0]
        jc = np.nonzero(target_ == Solver_.Names.core)[0]
        jx = np.intersect1d(ix, ja)

        # get correct input for quasi-newton solver
        def newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            ts = np.ones_like(tg) * np.nan
            tc = np.ones_like(tg) * np.nan

            ts[js] = Tmax[js]
            tc[js] = tg[js]

            ts[ja] = tg[ja]
            tc[ja] = 2 * Tmax[ja] - ts[ja]
            tc[jx] = (b[jx] * Tmax[jx] - a[jx] * ts[jx]) / (b[jx] - a[jx])

            tc[jc] = Tmax[jc]
            ts[jc] = tg[jc]

            return ts, tc

        return Tmax, newtheader

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

        Tmax, newtheader = self._steady_intensity_header(T, target)

        def balance(i: floatArray, tg: floatArray) -> floatArrayLike:
            ts, tc = newtheader(i, tg)
            return self.balance(ts, tc)

        def morgan(i: floatArray, tg: floatArray) -> floatArray:
            ts, tc = newtheader(i, tg)
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
        r = s.steady_intensity(Tmax, tol=1.0, maxiter=8, return_power=False)
        x, y, cnt, err = quasi_newton_2d(
            balance,
            morgan,
            r[Solver_.Names.transit].values,
            Tmax,
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
            ts, tc = newtheader(x, y)
            ta = self.average(ts, tc)

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
