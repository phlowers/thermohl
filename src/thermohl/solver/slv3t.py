from typing import Tuple, Type, Optional, Dict, Any

import numpy as np
import pandas as pd
from pyntb.optimize import qnewt2d_v

from thermohl import floatArrayLike, floatArray, strListLike, intArray
from thermohl.power.base import PowerTerm
from thermohl.solver.base import Solver as Solver_
from thermohl.solver.base import _DEFPARAM as DP
from thermohl.solver.base import _set_dates, reshape
from thermohl.solver.slv1t import Solver1T


def _profile_mom(
    ts: floatArrayLike, tc: floatArrayLike, r: floatArrayLike, re: floatArrayLike
) -> floatArrayLike:
    """Analytic temperature profile for steady heat equation in cylinder (mono-mat)."""
    return ts + (tc - ts) * (1.0 - (r / re) ** 2)


def _phi(r: floatArrayLike, ri: floatArrayLike, re: floatArrayLike) -> floatArrayLike:
    """Primitive function used in _profile_bim*** functions."""
    ri2 = ri**2
    return (0.5 * (r**2 - ri2) - ri2 * np.log(r / ri)) / (re**2 - ri2)


# TODO: Unused => to delete
# def _profile_bim(
#     ts: floatArrayLike,
#     tc: floatArrayLike,
#     r: floatArray,
#     ri: floatArrayLike,
#     re: floatArrayLike,
# ) -> floatArrayLike:
#     """Analytic temperature profile for steady heat equation in cylinder (bi-mat)."""
#     fl = lambda x: np.zeros_like(x)
#     fr = lambda x: (tc - ts) * _phi(x, ri, re) / _phi(re, ri, re)
#     return tc - np.piecewise(r, [r <= ri, r > ri], [fl, fr])


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

    @staticmethod
    def _morgan_coefficients(
        D: floatArrayLike, d: floatArrayLike, shape: Tuple[int, ...] = (1,)
    ) -> Tuple[floatArray, floatArray, floatArray, intArray]:
        """Coefficient for heat flux between surface and core in steady state."""
        c = 0.5 * np.ones(shape)
        D_ = D * np.ones_like(c)
        d_ = d * np.ones_like(c)
        i = np.nonzero(d_ > 0.0)[0]
        c[i] -= (d_[i] ** 2 / (D_[i] ** 2 - d_[i] ** 2)) * np.log(D_[i] / d_[i])
        # if len(shape) == 1 and shape[0] == 1:
        #     return c[0], D_[0], d_[0], i[0]
        # else:
        #     return c, D_, d_, i
        return c, D_, d_, i

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

    def update(self) -> None:
        self.args.extend_to_max_len()
        self.jh.__init__(**self.args.__dict__)
        self.sh.__init__(**self.args.__dict__)
        self.cc.__init__(**self.args.__dict__)
        self.rc.__init__(**self.args.__dict__)
        self.pc.__init__(**self.args.__dict__)

        self.mgc = Solver3T._morgan_coefficients(
            self.args.D, self.args.d, (self.args.max_len(),)
        )

        self.args.compress()

    def joule(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        # t is average temperature
        t = 0.5 * (ts + tc)
        c, D, d, ix = self.mgc
        t[ix] = _profile_bim_avg(ts[ix], tc[ix], 0.5 * d[ix], 0.5 * D[ix])
        return self.jh.value(t)

    def balance(self, ts: floatArray, tc: floatArray) -> floatArrayLike:
        return (
            self.joule(ts, tc)
            + self.sh.value(ts)
            - self.cc.value(ts)
            - self.rc.value(ts)
            - self.pc.value(ts)
        )

    def morgan(self, ts: floatArray, tc: floatArray) -> floatArray:
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

        # if no guess provided, use ambient temp
        shape = (self.args.max_len(),)
        if Tsg is None:
            Tsg = 1.0 * self.args.Ta
        if Tcg is None:
            Tcg = 1.5 * np.abs(self.args.Ta)
        Tsg_ = Tsg * np.ones(shape)
        Tcg_ = Tcg * np.ones(shape)

        # solve system
        x, y, cnt, err = qnewt2d_v(
            self.balance,
            self.morgan,
            Tsg_,
            Tcg_,
            rtol=tol,
            maxiter=maxiter,
            dx=1.0e-03,
            dy=1.0e-03,
        )
        if np.max(err) > tol or cnt == maxiter:
            print(f"rstat_analytic max err is {np.max(err):.3E} in {cnt:d} iterations")

        # format output
        z = 0.5 * (x + y)
        c, D, d, ix = self.mgc
        z[ix] = _profile_bim_avg(x[ix], y[ix], 0.5 * d[ix], 0.5 * D[ix])
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

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        Ts0: Optional[floatArrayLike] = None,
        Tc0: Optional[floatArrayLike] = None,
        transit: Optional[floatArrayLike] = None,
        Ta: Optional[floatArrayLike] = None,
        wind_speed: Optional[floatArrayLike] = None,
        wind_angle: Optional[floatArrayLike] = None,
        Pa: Optional[floatArrayLike] = None,
        rh: Optional[floatArrayLike] = None,
        pr: Optional[floatArrayLike] = None,
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
            raise ValueError()

        # get initial temperature
        if Ts0 is None:
            Ts0 = Ta
        if Tc0 is None:
            Tc0 = 1.0 + Ts0

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
        c, D, d, ix = self.mgc
        tpl = 2.0 * np.pi * self.args.l
        al = 0.5 * np.ones(n)
        a, b = _profile_bim_avg_coeffs(0.5 * d[ix], 0.5 * D[ix])
        al[ix] = a / b
        imc = 1.0 / (self.args.m * self.args.c)

        # init
        ts = np.zeros((N, n))
        ta = np.zeros((N, n))
        tc = np.zeros((N, n))
        ts[0, :] = Ts0
        tc[0, :] = Tc0
        ta[0, :] = 0.5 * (ts[0, :] + tc[0, :])

        # main time loop
        for i in range(1, len(time)):
            for k in de.keys():
                self.args[k] = de[k][i, :]
            self.update()
            bal = self.balance(ts[i, :], tc[i, :])
            ta[i, :] = ta[i - 1, :] + (time[i] - time[i - 1]) * bal * imc
            mrg = c * (self.jh.value(ta[i, :]) - bal) / tpl
            tc[i, :] = ta[i, :] + al * mrg
            ts[i, :] = tc[i, :] - mrg

        # save results
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
            for k in dr:
                if k == Solver_.Names.time:
                    continue
                dr[k] = dr[k][:, 0]

        return dr

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

        # save transit in arg
        transit = self.args.I

        # ...
        shape = (self.args.max_len(),)
        Tmax_ = T * np.ones(shape)

        # check target
        if target == "auto":
            target_ = None
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
                target_ = np.array([target for i in range(shape[0])])
        else:
            if len(target) != shape[0]:
                raise ValueError()
            for t in target:
                if t not in [Solver_.Names.surf, Solver_.Names.avg, Solver_.Names.core]:
                    raise ValueError()
            target_ = np.array(target)

        # pre-compute indexes
        c, D, d, ix = self.mgc
        a, b = _profile_bim_avg_coeffs(0.5 * d, 0.5 * D)

        if target_ is None:
            target_ = np.array([Solver_.Names.avg for i in range(shape[0])])
            target_[ix] = Solver_.Names.core

        js = np.nonzero(target_ == Solver_.Names.surf)[0]
        ja = np.nonzero(target_ == Solver_.Names.avg)[0]
        jc = np.nonzero(target_ == Solver_.Names.core)[0]
        jx = np.intersect1d(ix, ja)

        def _newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.I = i
            self.jh.__init__(**self.args.__dict__)
            ts = np.ones_like(tg) * np.nan
            tc = np.ones_like(tg) * np.nan

            ts[js] = Tmax_[js]
            tc[js] = tg[js]

            ts[ja] = tg[ja]
            tc[ja] = 2 * Tmax_[ja] - ts[ja]
            tc[jx] = (b[jx] * Tmax_[jx] - a[jx] * ts[jx]) / (b[jx] - a[jx])

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
        x, y, cnt, err = qnewt2d_v(
            balance,
            morgan,
            r[Solver_.Names.transit].values,
            Tmax_,
            rtol=tol,
            maxiter=maxiter,
            dx=1.0e-03,
            dy=1.0e-03,
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
            ta[ix] = _profile_bim_avg(ts[ix], tc[ix], 0.5 * d[ix], 0.5 * D[ix])

            if return_temp:
                df[Solver_.Names.tsurf] = ts
                df[Solver_.Names.tavg] = ta
                df[Solver_.Names.tcore] = tc

            if return_power:
                df[Solver_.Names.pjle] = self.joule(ts, tc)
                df[Solver_.Names.psol] = self.sh.value(ts)
                df[Solver_.Names.pcnv] = self.cc.value(ts)
                df[Solver_.Names.prad] = self.rc.value(ts)
                df[Solver_.Names.ppre] = self.pc.value(ts)

        # restore previous transit
        self.args.I = transit
        self.jh.__init__(**self.args.__dict__)

        return df
