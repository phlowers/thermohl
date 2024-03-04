"""Models to compute equilibrium temperature or max intensity in a conductor."""
import copy
import datetime
import time
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import thermohl.cigre as cig_
import thermohl.cner as cnr_
import thermohl.ieee as i3e_
import thermohl.olla as ola_
import thermohl.utils as utils
from pyntb.optimize import bisect_v
from thermohl.numeric import reshape


def default_values():
    """
    Get default values used in Solver class.

    Returns
    -------
    dict
        Dictionary of default values.

    """
    return utils.add_default_parameters({}, warning=False)


def _set_dates(month: Union[float, np.ndarray], day: Union[float, np.ndarray], hour: Union[float, np.ndarray],
               t: Union[float, np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set months, days and hours as 2D arrays.

    This function is used in transient temperature computations. Inputs month,
    day and hour are floats or 1D arrays of size n; input t is a time vector of
    size N with evaluation times in seconds. It sets arrays months, days and
    hours, of size (N, n) such that
        months[i, j] = datetime(month[j], day[j], hour[j]) + t[i] .
    """
    month2 = month * np.ones((n,), dtype=int)
    day2 = day * np.ones((n,), dtype=int)
    hour2 = hour * np.ones((n,), dtype=float)

    N = len(t)
    months = np.zeros((N, n), dtype=int)
    days = np.zeros((N, n), dtype=int)
    hours = np.zeros((N, n), dtype=float)

    td = np.array([datetime.timedelta()] + [datetime.timedelta(seconds=t[i] - t[i - 1]) for i in range(1, N)])

    for j in range(n):
        hj = int(np.floor(hour2[j]))
        dj = datetime.timedelta(seconds=3600. * (hour2[j] - hj))
        t0 = datetime.datetime(year=2000, month=month2[j], day=day2[j], hour=hj) + dj
        ts = pd.Series(t0 + td)
        months[:, j] = ts.dt.month
        days[:, j] = ts.dt.day
        hours[:, j] = ts.dt.hour + ts.dt.minute / 60. + (ts.dt.second + ts.dt.microsecond * 1.0E-06) / 3600.

    return months, days, hours


class Solver:
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and two cooling terms (convective and
    radiative cooling). This class is used to solve a temperature problem with
    the heating and cooling terms passed to its __init__ function.
    """

    def __init__(self, dc: dict,
                 jouleH: utils.PowerTerm, solarH: utils.PowerTerm,
                 convectiveC: utils.PowerTerm, radiativeC: utils.PowerTerm,
                 rainC: Optional[utils.PowerTerm] = None):
        """
        Create a Solver object.

        Parameters
        ----------
        dc : dict
            Input values used in power terms. If there is a missing value, a
            default is used.
        jouleH : utils.PowerTerm
            Joule heating term.
        solarH : utils.PowerTerm
            Solar heating term.
        convectiveC : utils.PowerTerm
            Convective cooling term.
        radiativeC : utils.PowerTerm
            Radiative cooling term.

        Returns
        -------
        None.

        """
        self.dc = copy.deepcopy(utils.add_default_parameters(dc))
        self.jh = jouleH()
        self.sh = solarH()
        self.cc = convectiveC()
        self.rc = radiativeC()
        if rainC is None:
            self.pc = utils.PowerTerm()
        else:
            self.pc = rainC()
        self.ctime = None
        return

    def _rhs_value(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (self.jh.value(T, **self.dc) + self.sh.value(T, **self.dc) -
                self.cc.value(T, **self.dc) - self.rc.value(T, **self.dc) -
                self.pc.value(T, **self.dc))

    def steady_temperature(self, Tmin: float = -99., Tmax: float = 999.,
                           tol: float = 1.0E-06, maxiter: int = 64,
                           return_core: bool = False, return_avg: bool = False, return_err: bool = False,
                           return_power: bool = True) -> pd.DataFrame:
        """
        Compute steady-state temperature.

        Parameters
        ----------
        Tmin : float, optional
            Lower bound for temperature. The default is -50.
        Tmax : float, optional
            Upper bound for temperature. The default is 300.
        tol : float, optional
            Tolerance for temperature error. The default is 1.0E-06.
        maxiter : int, optional
            Max number of iteration. The default is 64.
        return_core : bool, optional
            Return core temperature. Only valid with RTEm model. The default is False.
        return_avg : bool, optional
            Return average temperature. Only valid with RTE models. The default is False.
        return_err : bool, optional
            Return final error on temperature to check convergence. The default is False.
        return_power : bool, optional
            Return power term values. The default is True.

        Returns
        -------
        pandas.DataFrame
            A dataframe with temperature and other results (depending on inputs)
            in the columns.

        """
        t = time.time()

        # solve
        def _fun(x):
            return -self._rhs_value(x)

        T, err = bisect_v(_fun, Tmin, Tmax, (utils.dict_max_len(self.dc),), tol, maxiter)

        self.ctime = time.time() - t

        df = pd.DataFrame(data=T, columns=['T_surf'])

        if return_avg or return_core:
            try:
                Ta, Tc = self.jh._avg_core(T, **self.dc, xtol=tol, maxiter=maxiter)
            except AttributeError:
                Ta, Tc = np.nan, np.nan
            if return_avg:
                df['T_avg'] = Ta
            if return_core:
                df['T_core'] = Tc

        if return_err:
            df['err'] = err

        if return_power:
            df['P_joule'] = self.jh.value(T, **self.dc)
            df['P_solar'] = self.sh.value(T, **self.dc)
            df['P_convection'] = self.cc.value(T, **self.dc)
            df['P_radiation'] = self.rc.value(T, **self.dc)
            df['P_precipitation'] = self.pc.value(T, **self.dc)

        return df

    def transient_temperature(self, time: np.ndarray, T0: Optional[float] = None,
                              transit: Optional[np.ndarray] = None, Ta: Optional[np.ndarray] = None,
                              wind_angle: Optional[np.ndarray] = None, wind_speed: Optional[np.ndarray] = None,
                              Pa=None, rh=None, pr=None,
                              return_core: bool = False, return_avg: bool = False, return_power: bool = False) \
            -> pd.DataFrame:
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
        wind_angle : numpy.ndarray
            A 1D array with time-varying wind_angle. It should have the same size
            as input time. If set to None the value from internal dict will be
            used. The default is None.
        wind_speed : numpy.ndarray
            A 1D array with time-varying wind_speed. It should have the same size
            as input time. If set to None the value from internal dict will be
            used. The default is None.
        return_core : bool, optional
            Return core temperature. Only valid with RTEm model. The default is False.
        return_avg : bool, optional
            Return average temperature. Only valid with RTE models. The default is False.
        return_power : bool, optional
            Return power term values. The default is False.

        Returns
        -------
        pandas.DataFrame
            A dataframe with temperature and other results (depending on inputs)
            in the columns.

        """
        # if time-changing quantities are not provided, use ones from input
        # dict (static)
        if transit is None:
            transit = self.dc['I']
        if Ta is None:
            Ta = self.dc['Ta']
        if wind_angle is None:
            wind_angle = self.dc['wa']
        if wind_speed is None:
            wind_speed = self.dc['ws']
        if Pa is None:
            Pa = self.dc['Pa']
        if rh is None:
            rh = self.dc['rh']
        if pr is None:
            pr = self.dc['pr']

        # get sizes (n for input dict entries, N for time)
        n = utils.dict_max_len(self.dc)
        N = len(time)
        if N < 2:
            raise ValueError()

        # get initial temperature
        if T0 is None:
            T0 = Ta

        # get month, day and hours
        month, day, hour = _set_dates(self.dc['month'], self.dc['day'], self.dc['hour'], time, n)

        # Two dicts, one (dc) with static quantities (with all elements of size
        # n), the other (de) with time-changing quantities (with all elements of
        # size N*n); uk is a list of keys that are in dc but not in de.
        dc = utils.extend_to_max_len(self.dc, n=n)
        de = dict(month=month,
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
        uk = [k for k in dc.keys() if k not in de.keys()]

        # shortcuts for time-loop; T is the array where main temperature is stored
        m = self.dc['m']
        c = self.dc['c']
        T = np.zeros((N, n))
        T[0, :] = T0

        # main time loop
        for i in range(1, len(time)):
            for k in de.keys():
                dc[k] = de[k][i, :]
            q = T[i - 1, :]
            v = (self.jh.value(q, **dc) + self.sh.value(q, **dc) - self.cc.value(q, **dc) -
                 self.rc.value(q, **dc) - self.pc.value(q, **dc))
            T[i, :] = T[i - 1, :] + (time[i] - time[i - 1]) * v / (m * c)

        # manage return dict 1 : avg and core temperatures
        dr = dict(T_surf=T)
        if return_avg or return_core:
            Ta = np.zeros_like(T)
            Tc = np.zeros_like(T)
            try:
                dv = {}
                for j in range(n):
                    for k in uk:
                        dv[k] = dc[k][j]
                    for k in de.keys():
                        dv[k] = de[k][:, j]
                    Ta[:, j], Tc[:, j] = self.jh._avg_core(T[:, j], **dv)
            except AttributeError:
                Ta *= np.nan
                Tc *= np.nan
            if return_avg:
                dr['T_avg'] = Ta
            if return_core:
                dr['T_core'] = Tc

        # manage return dict 2 : powers
        if return_power:
            dv = {}
            for c in ['P_joule', 'P_solar', 'P_convection', 'P_radiation', 'P_precipitation']:
                dr[c] = np.zeros_like(T)
            for j in range(n):
                for k in uk:
                    dv[k] = dc[k][j]
                for k in de.keys():
                    dv[k] = de[k][:, j]
                dr['P_joule'][:, j] = self.jh.value(T[:, j], **dv)
                dr['P_solar'][:, j] = self.sh.value(T[:, j], **dv)
                dr['P_convection'][:, j] = self.cc.value(T[:, j], **dv)
                dr['P_radiation'][:, j] = self.rc.value(T[:, j], **dv)
                dr['P_precipitation'][:, j] = self.pc.value(T[:, j], **dv)

        # squeeze return values if n is 1
        if n == 1:
            for k in dr:
                dr[k] = dr[k][:, 0]

        return dr

    def steady_intensity(self, T: Union[float, np.ndarray], Imin: float = 0., Imax: float = 9999.,
                         target: str = 'surf', tol: float = 1.0E-06, maxiter: int = 64,
                         return_core: bool = False, return_avg: bool = False, return_surf: bool = False,
                         return_err: bool = False, return_power: bool = True) -> pd.DataFrame:
        """
        Compute steady-state max intensity.

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
        target : str, optional
            Target zone for maximum temperature. Possible values are surf (for surface),
            core (for core), and avg (for average). The default is surf.
        tol : float, optional
            Tolerance for temperature error. The default is 1.0E-06.
        maxiter : int, optional
            Max number of iteration. The default is 64.
        return_core : bool, optional
            Return core temperature. Only valid with RTEm model. The default is False.
        return_avg : bool, optional
            Return average temperature. Only valid with RTE models. The default is False.
        return_surf : bool, optional
            Return surface temperature. Only valid with RTE models. The default is False.
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
        tvalues = ['surf', 'core', 'avg']
        if target not in tvalues:
            raise ValueError('Wrong target value')

        t = time.time()
        n = utils.dict_max_len(self.dc)
        T *= np.ones((n,))
        I = self.dc['I']

        if target == 'surf':
            jh = (self.cc.value(T, **self.dc) + self.rc.value(T, **self.dc) +
                  self.pc.value(T, **self.dc) - self.sh.value(T, **self.dc))

            def fun(i):
                self.dc['I'] = i
                return jh - self.jh.value(T, **self.dc)
        elif target == 'core':
            def fun(i):
                self.dc['I'] = i
                Ta, Ts = self.jh._avg_surf(T, **self.dc)
                jh = (self.cc.value(Ts, **self.dc) + self.rc.value(Ts, **self.dc) +
                      self.pc.value(Ts, **self.dc) - self.sh.value(Ts, **self.dc))
                return -jh + self.jh.value(Ts, **self.dc)
        elif target == 'avg':
            def fun(i):
                self.dc['I'] = i
                Ts, Tc = self.jh._surf_core(T, **self.dc)
                jh = (self.cc.value(Ts, **self.dc) + self.rc.value(Ts, **self.dc) +
                      self.pc.value(Ts, **self.dc) - self.sh.value(Ts, **self.dc))
                return -jh + self.jh.value(Ts, **self.dc)

        A, e = bisect_v(fun, Imin, Imax, (n,), tol, maxiter)

        self.ctime = time.time() - t

        df = pd.DataFrame(data=A, columns=['I_max'])

        if return_surf or return_avg or return_core or return_power:
            self.dc['I'] = A
            if target == 'surf':
                Ts = T
                try:
                    Ta, Tc = self.jh._avg_core(T, **self.dc)
                except AttributeError:
                    Ta, Tc = np.nan, np.nan
            elif target == 'core':
                Tc = T
                Ta, Ts = self.jh._avg_surf(T, **self.dc)
            elif target == 'avg':
                Ta = T
                Ts, Tc = self.jh._surf_core(T, **self.dc)

            if return_surf:
                df.loc[:, 'T_surf'] = Ts
            if return_avg:
                df.loc[:, 'T_avg'] = Ta
            if return_core:
                df.loc[:, 'T_core'] = Tc

        if return_power:
            df['P_joule'] = self.jh.value(Ts, **self.dc)
            df['P_solar'] = self.sh.value(Ts, **self.dc)
            df['P_convection'] = self.cc.value(Ts, **self.dc)
            df['P_radiation'] = self.rc.value(Ts, **self.dc)
            df['P_precipitation'] = self.pc.value(T, **self.dc)

        if return_err:
            df['err'] = e

        self.dc['I'] = I

        return df


def cigre(dct: dict = {}):
    """
    Get a Solver using CIGRE models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, cig_.JouleHeating, cig_.SolarHeating,
                  cig_.ConvectiveCooling, cig_.RadiativeCooling)


def ieee(dct: dict = {}):
    """
    Get a Solver using IEEE models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, i3e_.JouleHeating, i3e_.SolarHeating,
                  i3e_.ConvectiveCooling, i3e_.RadiativeCooling)


def olla(dct: dict = {}, multi: bool = False):
    """
    Get a Solver using RTE-olla models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.
    multi: bool, optional
        Use multi-temp model within bisection iterations. The default is False.

    """
    if multi:
        return Solver(dct, ola_.JouleHeatingMulti, ola_.SolarHeating,
                      ola_.ConvectiveCooling, ola_.RadiativeCooling)
    else:
        return Solver(dct, ola_.JouleHeating, ola_.SolarHeating,
                      ola_.ConvectiveCooling, ola_.RadiativeCooling)


def cner(dct: dict = {}):
    """
    Get a Solver using RTE-cner models.

    Parameters
    ----------
    dct : dict, optional
        Input values. The default is {}.

    """
    return Solver(dct, cnr_.JouleHeating, cnr_.SolarHeating,
                  cnr_.ConvectiveCooling, cnr_.RadiativeCooling)
