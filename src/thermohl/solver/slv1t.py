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

from thermohl import floatArrayLike, floatArray
from thermohl.solver.base import Solver as Solver_
from thermohl.solver.base import _DEFPARAM as DP
from thermohl.solver.base import _set_dates, reshape
from thermohl.utils import bisect_v


class Solver1T(Solver_):
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

        Parameters
        ----------
        Tmin : float, optional
            Lower bound for temperature.
        Tmax : float, optional
            Upper bound for temperature.
        tol : float, optional
            Tolerance for temperature error.
        maxiter : int, optional
            Max number of iteration.
        return_err : bool, optional
            Return final error on temperature to check convergence. The default is False.
        return_power : bool, optional
            Return power term values. The default is True.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with temperature and other results (depending on inputs)
            in the columns.

        """

        # solve with bisection
        T, err = bisect_v(
            lambda x: -self.balance(x), Tmin, Tmax, (self.args.max_len(),), tol, maxiter
        )

        # format output
        df = pd.DataFrame(data=T, columns=[Solver_.Names.temp])

        if return_err:
            df[Solver_.Names.err] = err

        if return_power:
            df[Solver_.Names.pjle] = self.jh.value(T)
            df[Solver_.Names.psol] = self.sh.value(T)
            df[Solver_.Names.pcnv] = self.cc.value(T)
            df[Solver_.Names.prad] = self.rc.value(T)
            df[Solver_.Names.ppre] = self.pc.value(T)

        return df

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        T0: Optional[float] = None,
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
        T0 : float
            Initial temperature. If set to None, the ambient temperature from
            internal dict will be used. The default is None.
        return_power : bool, optional
            Return power term values. The default is False.

        Returns : Dict[str, Any]
            A dictionary with temperature and other results (depending on inputs)
            in the keys.
        """

        # get sizes (n for input dict entries, N for time)
        n = self.args.max_len()
        N = len(time)
        if N < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get initial temperature
        if T0 is None:
            T0 = (
                self.args.Ta
                if isinstance(self.args.Ta, numbers.Number)
                else self.args.Ta[0]
            )

        # get month, day and hours
        month, day, hour = _set_dates(
            self.args.month, self.args.day, self.args.hour, time, n
        )

        # Two dicts, one (dc) with static quantities (with all elements of size n), the other (de)
        # with time-changing quantities (with all elements of
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
        imc = 1.0 / (self.args.m * self.args.c)

        # init
        T = np.zeros((N, n))
        T[0, :] = T0

        # main time loop
        for i in range(1, len(time)):
            for k, v in de.items():
                self.args[k] = v[i, :]
            self.update()
            T[i, :] = (
                T[i - 1, :] + (time[i] - time[i - 1]) * self.balance(T[i - 1, :]) * imc
            )

        # save results
        dr = dict(time=time, T=T)

        # manage return dict 2 : powers
        if return_power:
            for c in Solver_.Names.powers():
                dr[c] = np.zeros_like(T)
            for i in range(N):
                for k in de.keys():
                    self.args[k] = de[k][i, :]
                self.update()
                dr[Solver_.Names.pjle][i, :] = self.jh.value(T[i, :])
                dr[Solver_.Names.psol][i, :] = self.sh.value(T[i, :])
                dr[Solver_.Names.pcnv][i, :] = self.cc.value(T[i, :])
                dr[Solver_.Names.prad][i, :] = self.rc.value(T[i, :])
                dr[Solver_.Names.ppre][i, :] = self.pc.value(T[i, :])

        # squeeze return values if n is 1
        if n == 1:
            keys = list(dr.keys())
            keys.remove(Solver_.Names.time)
            for k in keys:
                dr[k] = dr[k][:, 0]

        return dr

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

        Parameters
        ----------
        T : float or numpy.ndarray
            Maximum temperature.
        Imin : float, optional
            Lower bound for intensity. The default is 0.
        Imax : float, optional
            Upper bound for intensity. The default is 9999.
        tol : float, optional
            Tolerance for temperature error. The default is 1.0E-06.
        maxiter : int, optional
            Max number of iteration. The default is 64.
        return_err : bool, optional
            Return final error on intensity to check convergence. The default is False.
        return_power : bool, optional
            Return power term values. The default is True.

        Returns
        -------
        pandas.DataFrame
            A dataframe with maximum intensity and other results (depending on inputs)
            in the columns.

        """

        # save transit in arg
        transit = self.args.I

        # solve with bisection
        shape = (self.args.max_len(),)
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

        if return_err:
            df[Solver_.Names.err] = err

        if return_power:
            df[Solver_.Names.pjle] = self.jh.value(T)
            df[Solver_.Names.psol] = self.sh.value(T)
            df[Solver_.Names.pcnv] = self.cc.value(T)
            df[Solver_.Names.prad] = self.rc.value(T)
            df[Solver_.Names.ppre] = self.pc.value(T)

        return df
