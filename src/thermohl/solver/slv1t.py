# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numbers
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from pyntb.optimize import bisect_v

from thermohl import floatArrayLike, floatArray
from thermohl.solver.base import Solver as Solver_, Args
from thermohl.solver.base import _DEFPARAM as DP


class Solver1T(Solver_):

    def _steady_return_opt(
        self,
        return_err: bool,
        return_power: bool,
        T: np.ndarray,
        err: np.ndarray,
        df: pd.DataFrame,
    ):
        """Add error and/or power values to pd.Dataframe returned in
        steady_temperature and steady_intensity methods."""

        # add convergence error if asked
        if return_err:
            df[Solver_.Names.err] = err

        # add power values if asked
        if return_power:
            df[Solver_.Names.pjle] = self.jh.value(T)
            df[Solver_.Names.psol] = self.sh.value(T)
            df[Solver_.Names.pcnv] = self.cc.value(T)
            df[Solver_.Names.prad] = self.rc.value(T)
            df[Solver_.Names.ppre] = self.pc.value(T)

        return df

    def steady_temperature(
        self,
        Tmin: float = DP.tmin,
        Tmax: float = DP.tmax,
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """
        Compute steady-state temperature.

        Args:
            Tmin (float, optional): Lower bound for temperature.
            Tmax (float, optional): Upper bound for temperature.
            tol (float, optional): Tolerance for temperature error.
            maxiter (int, optional): Max number of iterations.
            return_err (bool, optional): Return final error on temperature to check convergence. The default is False.
            return_power (bool, optional): Return power term values. The default is True.

        Returns:
            pandas.DataFrame: A DataFrame with temperature and other results (depending on inputs) in the columns.

        """

        # solve with bisection
        T, err = bisect_v(
            lambda x: -self.balance(x),
            Tmin,
            Tmax,
            self._min_shape(),
            tol=tol,
            maxiter=maxiter,
        )

        # format output
        df = pd.DataFrame(data=T, columns=[Solver_.Names.temp])
        df = self._steady_return_opt(return_err, return_power, T, err, df)

        return df

    def steady_intensity(
        self,
        T: floatArrayLike = np.array([]),
        Imin: float = DP.imin,
        Imax: float = DP.imax,
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """Compute steady-state max intensity.

        Compute the maximum intensity that can be run in a conductor without
        exceeding the temperature given in argument.

        Args:
            T (float | numpy.ndarray): Maximum temperature.
            Imin (float, optional): Lower bound for intensity. The default is 0.
            Imax (float, optional): Upper bound for intensity. The default is 9999.
            tol (float, optional): Tolerance for temperature error. The default is 1.0E-06.
            maxiter (int, optional): Max number of iterations. The default is 64.
            return_err (bool, optional): Return final error on intensity to check convergence. The default is False.
            return_power (bool, optional): Return power term values. The default is True.

        Returns:
            pandas.DataFrame: A dataframe with maximum intensity and other results (depending on inputs) in the columns.

        """

        # save transit in arg
        transit = self.args.I

        # solve with bisection
        shape = self._min_shape()
        T_ = T * np.ones(shape)
        jh = (
            self.cc.value(T_)
            + self.rc.value(T_)
            + self.pc.value(T_)
            - self.sh.value(T_)
        )

        def fun(i: floatArray) -> floatArrayLike:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            return self.jh.value(T_) - jh

        A, err = bisect_v(fun, Imin, Imax, shape, tol, maxiter)

        # restore previous transit
        self.args.I = transit

        # format output
        df = pd.DataFrame(data=A, columns=[Solver_.Names.transit])
        df = self._steady_return_opt(return_err, return_power, T_, err, df)

        return df

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        T0: Optional[float] = None,
        dynamic: dict = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

        Args:
            time (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            T0 (float | None): Initial temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs) in the keys.
        """

        # get sizes (n for input dict entries, N for time)
        n = self._min_shape()[0]
        N = len(time)

        # process dynamic values
        dynamic_ = self._transient_process_dynamic(time, n, dynamic)

        # shortcuts for time-loop
        imc = 1.0 / (self.args.m * self.args.c)

        # save args
        args = self.args.__dict__.copy()

        # initial conditions
        T = np.zeros((N, n))
        if T0 is None:
            T0 = self.args.Ta
        T[0, :] = T0

        # time loop
        for i in range(1, N):
            for k, v in dynamic_.items():
                self.args[k] = v[i, :]
            self.update()
            T[i, :] = (
                T[i - 1, :] + (time[i] - time[i - 1]) * self.balance(T[i - 1, :]) * imc
            )

        # save results
        dr = {Solver_.Names.time: time, Solver_.Names.temp: T}

        # add power to return dict if needed
        if return_power:
            for c in Solver_.Names.powers():
                dr[c] = np.zeros_like(T)
            for i in range(N):
                for k, v in dynamic_.items():
                    self.args[k] = v[i, :]
                self.update()
                dr[Solver_.Names.pjle][i, :] = self.jh.value(T[i, :])
                dr[Solver_.Names.psol][i, :] = self.sh.value(T[i, :])
                dr[Solver_.Names.pcnv][i, :] = self.cc.value(T[i, :])
                dr[Solver_.Names.prad][i, :] = self.rc.value(T[i, :])
                dr[Solver_.Names.ppre][i, :] = self.pc.value(T[i, :])

        # squeeze values in return dict (if n is 1)
        if n == 1:
            keys = list(dr.keys())
            keys.remove(Solver_.Names.time)
            for k in keys:
                dr[k] = dr[k][:, 0]

        # restore args
        self.args = Args(args)

        return dr
