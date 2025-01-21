import numbers
from array import array
from copy import deepcopy
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from mypy.server import update
from numpy import number
from pyntb.optimize import bisect_v

from thermohl import floatArrayLike, floatArray
from thermohl.solver.base import Args
from thermohl.solver.base import Solver as Solver_
from thermohl.solver.base import _DEFPARAM as DP
from thermohl.solver.base import _set_dates, reshape


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
            lambda x: -self.balance(x),
            Tmin,
            Tmax,
            (self.args.max_len(),),
            tol,
            maxiter,
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
        transit: Optional[floatArray] = None,
        Ta: Optional[floatArray] = None,
        wind_speed: Optional[floatArray] = None,
        wind_angle: Optional[floatArray] = None,
        Pa: Optional[floatArray] = None,
        rh: Optional[floatArray] = None,
        pr: Optional[floatArray] = None,
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
        transit : numpy.ndarray
            A 1D array with time-varying transit. It should have the same size
            as input time. If set to None the value from internal dict will be
            used. The default is None.
        Ta : numpy.ndarray
            A 1D array with time-varying ambient temperature. It should have the
            same size as input time. If set to None the value from internal dict
            will be used. The default is None.
        wind_speed : numpy.ndarray
            A 1D array with time-varying wind_speed. It should have the same size
            as input time. If set to None the value from internal dict will be
            used. The default is None.
        wind_angle : numpy.ndarray
            A 1D array with time-varying wind_angle. It should have the same size
            as input time. If set to None the value from internal dict will be
            used. The default is None.
        Pa : numpy.ndarray
            A 1D array with time-varying atmospheric pressure. It should have the
            same size as input time. If set to None the value from internal dict
            will be used. The default is None.
        rh : numpy.ndarray
            A 1D array with time-varying relative humidity. It should have the
            same size as input time. If set to None the value from internal dict
            will be used. The default is None.
        pr : numpy.ndarray
            A 1D array with time-varying precipitation. It should have the
            same size as input time. If set to None the value from internal dict
            will be used. The default is None.
        return_power : bool, optional
            Return power term values. The default is False.

        Returns
        -------
        Dict[str, Any]
            A dictionary with temperature and other results (depending on inputs)
            in the keys.

        """

        # if time-changing quantities are not provided, use ones from args (static)
        transit = transit if transit is not None else self.args.I
        Ta = Ta if Ta is not None else self.args.Ta
        wind_speed = wind_speed if wind_speed is not None else self.args.ws
        wind_angle = wind_angle if wind_angle is not None else self.args.wa
        Pa = Pa if Pa is not None else self.args.Pa
        rh = rh if rh is not None else self.args.rh
        pr = pr if pr is not None else self.args.pr

        # get sizes (n for input dict entries, N for time)
        n = self.args.max_len()
        N = len(time)
        if N < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get initial temperature
        if T0 is None:
            T0 = Ta if isinstance(Ta, numbers.Number) else Ta[0]

        # get month, day and hours
        month, day, hour = _set_dates(
            self.args.month, self.args.day, self.args.hour, time, n
        )

        # save args
        args = self.args.__dict__.copy()

        # Two dicts, one (dc) with static quantities (with all elements of size
        # n), the other (de) with time-changing quantities (with all elements of
        # size N*n); uk is a list of keys that are in dc but not in de.
        de = dict(
            month=month,
            day=day,
            hour=hour,
            I=reshape(transit, N, n),
            Ta=reshape(Ta, N, n),
            wa=reshape(wind_angle, N, n),
            ws=reshape(wind_speed, N, n),
            Pa=reshape(Pa, N, n),
            rh=reshape(rh, N, n),
            pr=reshape(pr, N, n),
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
            for k in dr:
                if k == Solver_.Names.time:
                    continue
                dr[k] = dr[k][:, 0]

        # restore args
        self.args = Args(args)

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
