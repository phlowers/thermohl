from typing import Tuple, Type, Union

import numpy as np
import pandas as pd
from pyntb.optimize import qnewt2d_v

from thermohl.power.base import PowerTerm
from thermohl.solver.base import Solver as Solver_
from thermohl.solver.base import _DEFPARAM as DP


def _profile_mom(ts: float, tc: float, r: Union[float, np.ndarray], re: float) -> Union[float, np.ndarray]:
    """Analytic temperature profile for steady heat equation in cylinder (mono-mat)."""
    return ts + (tc - ts) * (1. - (r / re)**2)


def _phi(r: Union[float, np.ndarray], ri: float, re: float) -> Union[float, np.ndarray]:
    """Primitive function used in _profile_bim*** functions."""
    ri2 = ri**2
    return (0.5 * (r**2 - ri2) - ri2 * np.log(r / ri)) / (re**2 - ri2)


def _profile_bim(ts: float, tc: float, r: Union[float, np.ndarray], ri: float, re: float) -> Union[float, np.ndarray]:
    """Analytic temperature profile for steady heat equation in cylinder (bi-mat)."""
    fl = lambda x: np.zeros_like(x)
    fr = lambda x: (tc - ts) * _phi(x, ri, re) / _phi(re, ri, re)
    return tc - np.piecewise(r, [r <= ri, r > ri], [fl, fr])


def _profile_bim_avg(ts: float, tc: float, ri: float, re: float) -> float:
    """Analytical formulation for average temperature in _profile_bim."""
    ri2 = ri**2
    re2 = re**2
    a = 0.5 * (re2 - ri2)**2 - re2 * ri2 * (2. * np.log(re / ri) - 1.) - ri**4
    b = 2. * re2 * (re2 - ri2) * _phi(re, ri, re)
    return tc - (a / b) * (tc - ts)


# def rsolve_simplified(t, I, Ts0, Tc0, slv):
#     pp = dict(
#         t=t,
#         Tsurf=np.zeros_like(t),
#         Tcore=np.zeros_like(t),
#         Tavg=np.zeros_like(t),
#         Pjou=np.zeros_like(t),
#         Psol=np.zeros_like(t),
#         Pcnv=np.zeros_like(t),
#         Prad=np.zeros_like(t),
#         Ppre=np.zeros_like(t),
#     )
#
#     def powers(ta, ts, I_):
#         pj = _JouleHeating.value(ta, I_, slv.dc['D'], slv.dc['d'], slv.dc['A'], slv.dc['a'], slv.dc['km'],
#                                  slv.dc['ki'], slv.dc['kl'], slv.dc['kq'], slv.dc['RDC20'])
#         ps = slv.sh.value(ts, **slv.dc)
#         pc = slv.cc.value(np.array([ts]), **slv.dc)[0]
#         pr = slv.rc.value(np.array(ts), **slv.dc)
#         pp = slv.pc.value(np.array([ts]), **slv.dc)[0]
#         return pj, ps, pc, pr, pp
#
#     # pre-comp
#     c = _morgan_coeff(slv.dc['D'], slv.dc['d'], (1,))
#     rho = slv.dc['m'] / slv.dc['A']
#     cp = slv.dc['c']
#     re = 0.5 * slv.dc['D']
#     ri = 0.5 * slv.dc['d']
#     re2 = re**2
#     ri2 = ri**2
#     den = rho * cp * np.pi * re2
#     tpl = 2. * np.pi * slv.dc['l']
#     if ri:
#         a = 0.5 * (re2 - ri2)**2 - re2 * ri2 * (2. * np.log(re / ri) - 1.) - ri**4
#         b = 2. * re2 * (re2 - ri2) * _phi(re, ri, re)
#         al = a / b
#     else:
#         al = 0.5
#
#     # init
#     pp['Tsurf'][0] = Ts0
#     pp['Tcore'][0] = Tc0
#     if ri > 0.:
#         pp['Tavg'][0] = _profile_bim_avg(Ts0, Tc0, ri, re)
#     else:
#         pp['Tavg'][0] = 0.5 * (Ts0 + Tc0)
#
#     pj, ps, pc, pr, pp_ = powers(pp['Tavg'][0], pp['Tsurf'][0], I[0])
#     pb = pj + ps - pc - pr - pp_
#     pp['Pjou'][0] = pj
#     pp['Psol'][0] = ps
#     pp['Pcnv'][0] = pc
#     pp['Prad'][0] = pr
#     pp['Ppre'][0] = pp_
#
#     for i in range(len(t) - 1):
#         ta = pp['Tavg'][i] + (t[i + 1] - t[i]) * pb / den
#         mg = c * (pp['Pjou'][i] - pb) / tpl
#         tc = ta + al * mg
#         ts = tc - mg
#
#         pj, ps, pc, pr, pp_ = powers(ta, ts, I[i])
#         pb = pj + ps - pc - pr - pp_
#
#         pp['Tsurf'][i + 1] = ts
#         pp['Tavg'][i + 1] = ta
#         pp['Tcore'][i + 1] = tc
#         pp['Pjou'][i + 1] = pj
#         pp['Psol'][i + 1] = ps
#         pp['Pcnv'][i + 1] = pc
#         pp['Prad'][i + 1] = pr
#         pp['Ppre'][i + 1] = pp_
#
#     return pp


class Solver3T(Solver_):

    @staticmethod
    def _morgan_coefficients(
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            shape: Tuple[int, ...] = (1,)
    ):
        """Coefficient for heat flux between surface and core in steady state."""
        c = 0.5 * np.ones(shape)
        D_ = D * np.ones_like(c)
        d_ = d * np.ones_like(c)
        i = d_ > 0.
        c[i] -= (d_[i]**2 / (D_[i]**2 - d_[i]**2)) * np.log(D_[i] / d_[i])
        if len(shape) == 1 and shape[0] == 1:
            return c[0], D_[0], d_[0], i[0]
        else:
            return c, D_, d_, i

    def __init__(
            self,
            dic: dict = None,
            joule: Type[PowerTerm] = PowerTerm,
            solar: Type[PowerTerm] = PowerTerm,
            convective: Type[PowerTerm] = PowerTerm,
            radiative: Type[PowerTerm] = PowerTerm,
            precipitation: Type[PowerTerm] = PowerTerm
    ):
        super().__init__(dic, joule, solar, convective, radiative, precipitation)
        self.update()

    def update(self):
        self.args.extend_to_max_len(inplace=True)
        self.jh.__init__(**self.args.__dict__)
        self.sh.__init__(**self.args.__dict__)
        self.cc.__init__(**self.args.__dict__)
        self.rc.__init__(**self.args.__dict__)
        self.pc.__init__(**self.args.__dict__)

        self.mgc = Solver3T._morgan_coefficients(self.args.D, self.args.d, self.args.max_len())

        self.args.compress()
        return

    def joule(self, ts: Union[float, np.ndarray], tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # t is average temperature
        t = 0.5 * (ts + tc)
        c, D, d, ix = self.mgc
        t[ix] = _profile_bim_avg(ts[ix], tc[ix], 0.5 * d[ix], 0.5 * D[ix])
        return self.jh.value(t)

    def balance(self, ts: Union[float, np.ndarray], tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (
                self.joule(ts, tc) +
                self.sh.value(ts) -
                self.cc.value(ts) -
                self.rc.value(ts) -
                self.pc.value(ts)
        )

    def morgan(self, ts: Union[float, np.ndarray], tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        c, _, _, _ = self.mgc
        return (tc - tc) - c * self.joule(ts, tc) / (2. * np.pi * self.args.l)

    def steady_temperature(
            self,
            Tsg=None,
            Tcg=None,
            tol: float = DP.tol,
            maxiter: int = DP.maxiter,
            return_err: bool = False,
            return_temp: bool = False,
            return_power: bool = True):

        # if no guess provided, use ambient temp
        shape = (self.args.max_len(),)
        if Tsg is None:
            Tsg = 1. * self.args.Ta
        if Tcg is None:
            Tcg = 1.5 * np.abs(self.args.Ta)
        Tsg_ = Tsg * np.ones(shape)
        Tcg_ = Tcg * np.ones(shape)

        # solve system
        x, y, i, e = qnewt2d_v(self.balance, self.morgan, Tsg_, Tcg_, rtol=tol, maxiter=maxiter, dx=1.0E-03,
                               dy=1.0E-03)
        if np.max(e) > tol or i == maxiter:
            print(f"rstat_analytic max err is {np.max(e):.3E} in {i:d} iterations")

        # format output
        z = 0.5 * (x + y)
        c, D, d, ix = self.mgc
        z[ix] = _profile_bim_avg(x[ix], y[ix], 0.5 * d[ix], 0.5 * D[ix])
        df = pd.DataFrame({Solver_.Names.tsurf: x, Solver_.Names.tavg: z, Solver_.Names.tcore: y})

        if return_err:
            df[Solver_.Names.err] = e

        if return_power:
            df[Solver_.Names.pjle] = self.joule(x, y)
            df[Solver_.Names.psol] = self.sh.value(x)
            df[Solver_.Names.pcnv] = self.cc.value(x)
            df[Solver_.Names.prad] = self.rc.value(x)
            df[Solver_.Names.ppre] = self.pc.value(x)

    def transient_temperature(self):
        pass

    def steady_intensity(
            self,
            Tmax: Union[float, np.ndarray],
            target,
            tol: float = DP.tol,
            maxiter: int = DP.maxiter,
            return_err: bool = False,
            return_power: bool = True
    ):
        pass
