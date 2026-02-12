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


def _profile_mom(
    surface_temperature_c: float,
    core_temperature_c: float,
    r: floatArrayLike,
    re: float,
) -> floatArrayLike:
    """Analytic temperature profile for steady heat equation in cylinder (mono-mat)."""
    return surface_temperature_c + (core_temperature_c - surface_temperature_c) * (
        1.0 - (r / re) ** 2
    )


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
    surface_temperature_c: floatArrayLike,
    core_temperature_c: floatArrayLike,
    ri: floatArrayLike,
    re: floatArrayLike,
) -> floatArrayLike:
    """Analytical formulation for average temperature in _profile_bim."""
    a, b = _profile_bim_avg_coeffs(ri, re)
    return core_temperature_c - (a / b) * (core_temperature_c - surface_temperature_c)


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
            - heat_capacity_jkgk : numpy.ndarray[float]
                Coefficient array for heat flux.
            - D_ : numpy.ndarray[float]
                Array of core diameters, broadcasted to the shape of `heat_capacity_jkgk`.
            - d_ : numpy.ndarray[float]
                Array of surface diameters, broadcasted to the shape of `heat_capacity_jkgk`.
            - i : numpy.ndarray[int]
                Indices where surface diameter `d_` is greater than 0.
        """
        heat_capacity_jkgk = 0.5 * np.ones((self.args.max_len(),))
        outer_diameter_m = self.args.outer_diameter_m * np.ones_like(heat_capacity_jkgk)
        core_diameter_m = self.args.core_diameter_m * np.ones_like(heat_capacity_jkgk)
        i = np.nonzero(core_diameter_m > 0.0)[0]
        heat_capacity_jkgk[i] -= (
            core_diameter_m[i] ** 2
            / (outer_diameter_m[i] ** 2 - core_diameter_m[i] ** 2)
        ) * np.log(outer_diameter_m[i] / core_diameter_m[i])
        return heat_capacity_jkgk, outer_diameter_m, core_diameter_m, i

    def update(self) -> None:
        """
        Updates the solver's internal state by reinitializing several components
        and recalculating the Morgan coefficients.
        This method performs the following steps:
        1. Extends the arguments to their maximum length.
        2. Reinitializes the `joule_heating`, `solar_heating`, `convective_cooling`, `radiative_cooling`, and `precipitation_cooling` components using the updated arguments.
        3. Recalculates the Morgan coefficients using the updated dimensions.
        4. Compresses the arguments.
        Returns:
            None
        """
        self.args.extend_to_max_len()
        self.joule_heating.__init__(**self.args.__dict__)
        self.solar_heating.__init__(**self.args.__dict__)
        self.convective_cooling.__init__(**self.args.__dict__)
        self.radiative_cooling.__init__(**self.args.__dict__)
        self.precipitation_cooling.__init__(**self.args.__dict__)

        self.morgan_coefficients = self._morgan_coefficients()

        self.args.compress()

    def average(
        self, surface_temperature_c: floatArray, core_temperature_c: floatArray
    ) -> floatArrayLike:
        """
        Compute average temperature given surface and core temperature.

        This formula is based on analytical solution in steady-state mode. For
        single material, the formula reduces itself to an usual mean; for
        bi-material conductors, we have geometrical terms to take into account.

        Args:
            surface_temperature_c (numpy.ndarray): Array of surface temperatures.
            core_temperature_c (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: Array of average temperatures.
        """
        ambient_temperature_c = 0.5 * (surface_temperature_c + core_temperature_c)
        _, outer_diameter_m, core_diameter_m, ix = self.morgan_coefficients
        ambient_temperature_c[ix] = _profile_bim_avg(
            surface_temperature_c[ix],
            core_temperature_c[ix],
            0.5 * core_diameter_m[ix],
            0.5 * outer_diameter_m[ix],
        )
        return ambient_temperature_c

    def joule(
        self, surface_temperature_c: floatArray, core_temperature_c: floatArray
    ) -> floatArrayLike:
        """
        Calculate the Joule heating effect.

        Args:
            surface_temperature_c (numpy.ndarray): Array of surface temperatures.
            core_temperature_c (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: The calculated Joule heating values.

        Notes:
        - The function computes the average temperature `temperature`.
        - Returns the Joule heating values based on the adjusted temperatures.
        """
        ambient_temperature_c = self.average(surface_temperature_c, core_temperature_c)
        return self.joule_heating.value(ambient_temperature_c)

    def balance(
        self, surface_temperature_c: floatArray, core_temperature_c: floatArray
    ) -> floatArrayLike:
        """
        Calculate the thermal balance.

        This method computes the thermal balance by summing the joule heating,
        specific heat, and subtracting the contributions from the cooling
        components (convection, radiation, and conduction).

        Args:
            surface_temperature_c (numpy.ndarray): Array of surface temperatures.
            core_temperature_c (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: The resulting thermal balance.
        """
        return (
            self.joule(surface_temperature_c, core_temperature_c)
            + self.solar_heating.value(surface_temperature_c)
            - self.convective_cooling.value(surface_temperature_c)
            - self.radiative_cooling.value(surface_temperature_c)
            - self.precipitation_cooling.value(surface_temperature_c)
        )

    def morgan(
        self, surface_temperature_c: floatArray, core_temperature_c: floatArray
    ) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Args:
            surface_temperature_c (numpy.ndarray): Array of surface temperatures.
            core_temperature_c (numpy.ndarray): Array of core temperatures.

        Returns:
            numpy.ndarray: Resulting array after applying the Morgan function.
        """
        heat_capacity_jkgk, _, _, _ = self.morgan_coefficients
        return (
            core_temperature_c - surface_temperature_c
        ) - heat_capacity_jkgk * self.joule(
            surface_temperature_c, core_temperature_c
        ) / (2.0 * np.pi * self.args.radial_thermal_conductivity_wmk)

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

        Args:
            Tsg (float | numpy.ndarray | None): Initial guess for the surface temperature. If None, ambient temperature is used.
            Tcg (float | numpy.ndarray | None): Initial guess for the core temperature. If None, 1.5 times the absolute value of ambient temperature is used.
            tol (float): Tolerance for the quasi-Newton solver.
            maxiter (int): Maximum number of iterations for the quasi-Newton solver.
            return_err (bool): If True, the error of the solution is included in the returned DataFrame.
            return_power (bool): If True, power-related values are included in the returned DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the steady-state temperatures and optionally the error and power-related values.
        """

        # if no guess provided, use ambient temp
        shape = (self.args.max_len(),)
        Tsg = Tsg if Tsg is not None else 1.0 * self.args.ambient_temperature_c
        Tcg = Tcg if Tcg is not None else 1.5 * np.abs(self.args.ambient_temperature_c)
        Tsg_ = Tsg * np.ones(shape)
        Tcg_ = Tcg * np.ones(shape)

        # solve system
        x, y, cnt, err = quasi_newton_2d(
            func1=self.balance,
            func2=self.morgan,
            x_init=Tsg_,
            y_init=Tcg_,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or cnt == maxiter:
            print(
                f"rstat_analytic max err is {np.max(err):.3E} in {cnt:core_diameter_m} iterations"
            )

        # format output
        z = self.average(x, y)
        df = pd.DataFrame(
            {Solver_.Names.tsurf: x, Solver_.Names.tavg: z, Solver_.Names.tcore: y}
        )

        if return_err:
            df[Solver_.Names.err] = err

        if return_power:
            df[Solver_.Names.pjle] = self.joule(x, y)
            df[Solver_.Names.psol] = self.solar_heating.value(x)
            df[Solver_.Names.pcnv] = self.convective_cooling.value(x)
            df[Solver_.Names.prad] = self.radiative_cooling.value(x)
            df[Solver_.Names.ppre] = self.precipitation_cooling.value(x)

        return df

    def _morgan_transient(self):
        """Morgan coefficients for transient temperature."""
        heat_capacity_jkgk, outer_diameter_m, core_diameter_m, ix = (
            self.morgan_coefficients
        )
        c1 = heat_capacity_jkgk / (
            2.0 * np.pi * self.args.radial_thermal_conductivity_wmk
        )
        c2 = 0.5 * np.ones_like(c1)
        a, b = _profile_bim_avg_coeffs(
            0.5 * core_diameter_m[ix], 0.5 * outer_diameter_m[ix]
        )
        c2[ix] = a / b
        return c1, c2

    def _transient_temperature_results(
        self,
        time,
        surface_temperature_c,
        ambient_temperature_c,
        core_temperature_c,
        return_power,
        n,
    ):
        dr = {
            Solver_.Names.time: time,
            Solver_.Names.tsurf: surface_temperature_c,
            Solver_.Names.tavg: ambient_temperature_c,
            Solver_.Names.tcore: core_temperature_c,
        }

        if return_power:
            for power in Solver_.Names.powers():
                dr[power] = np.zeros_like(surface_temperature_c)

            for i in range(len(time)):
                dr[Solver_.Names.pjle][i, :] = self.joule(
                    surface_temperature_c[i, :], core_temperature_c[i, :]
                )
                dr[Solver_.Names.psol][i, :] = self.solar_heating.value(
                    surface_temperature_c[i, :]
                )
                dr[Solver_.Names.pcnv][i, :] = self.convective_cooling.value(
                    surface_temperature_c[i, :]
                )
                dr[Solver_.Names.prad][i, :] = self.radiative_cooling.value(
                    surface_temperature_c[i, :]
                )
                dr[Solver_.Names.ppre][i, :] = self.precipitation_cooling.value(
                    surface_temperature_c[i, :]
                )

        if n == 1:
            keys = list(dr.keys())
            keys.remove(Solver_.Names.time)
            for k in keys:
                dr[k] = dr[k][:, 0]

        return dr

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        surface_temperature_0_c: Optional[floatArrayLike] = None,
        core_temperature_0_c: Optional[floatArrayLike] = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

        Args:
            time (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            surface_temperature_0_c (float | numpy.ndarray | None): Initial surface temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            core_temperature_0_c (float | numpy.ndarray | None): Initial core temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs) in the keys.

        """
        # get sizes (n for input dict entries, N for time)
        n = self.args.max_len()
        N = len(time)
        if N < 2:
            raise ValueError()

        # get initial temperature
        surface_temperature_0_c = (
            surface_temperature_0_c
            if surface_temperature_0_c is not None
            else self.args.ambient_temperature_c
        )
        core_temperature_0_c = (
            core_temperature_0_c
            if core_temperature_0_c is not None
            else 1.0 + surface_temperature_0_c
        )

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
            current_a=reshape(self.args.current_a, N, n),
            ambient_temperature_c=reshape(self.args.ambient_temperature_c, N, n),
            wind_angle_deg=reshape(self.args.wind_angle_deg, N, n),
            wind_speed_ms=reshape(self.args.wind_speed_ms, N, n),
            ambient_pressure_pa=reshape(self.args.ambient_pressure_pa, N, n),
            relative_humidity=reshape(self.args.relative_humidity, N, n),
            precipitation_rate_ms=reshape(self.args.precipitation_rate_ms, N, n),
        )
        del (month, day, hour)

        # shortcuts for time-loop
        c1, c2 = self._morgan_transient()
        imc = 1.0 / (self.args.linear_mass_kgm * self.args.heat_capacity_jkgk)

        # init
        surface_temperature_c = np.zeros((N, n))
        ambient_temperature_c = np.zeros((N, n))
        core_temperature_c = np.zeros((N, n))
        surface_temperature_c[0, :] = surface_temperature_0_c
        core_temperature_c[0, :] = core_temperature_0_c
        ambient_temperature_c[0, :] = self.average(
            surface_temperature_c[0, :], core_temperature_c[0, :]
        )

        # main time loop
        for i in range(1, len(time)):
            for k in de.keys():
                self.args[k] = de[k][i, :]
            self.update()
            bal = self.balance(
                surface_temperature_c[i - 1, :], core_temperature_c[i - 1, :]
            )
            ambient_temperature_c[i, :] = (
                ambient_temperature_c[i - 1, :] + (time[i] - time[i - 1]) * bal * imc
            )
            mrg = c1 * (self.joule_heating.value(ambient_temperature_c[i, :]) - bal)
            core_temperature_c[i, :] = ambient_temperature_c[i, :] + c2 * mrg
            surface_temperature_c[i, :] = core_temperature_c[i, :] - mrg

        return self._transient_temperature_results(
            time,
            surface_temperature_c,
            ambient_temperature_c,
            core_temperature_c,
            return_power,
            n,
        )

    @staticmethod
    def _check_target(target, core_diameter_m, max_len):
        """
        Validates and processes the target temperature input.

        Args:
            target (str | list[str]): The target temperature(s) to be validated. It can be:
                - "auto": which sets the target automatically.
                - A string: must be one of Solver_.Names.surf, Solver_.Names.avg, or Solver_.Names.core.
                - A list of strings: each string must be one of Solver_.Names.surf, Solver_.Names.avg, or Solver_.Names.core.
            max_len (int): The expected length of the target list if target is a list.

        Returns:
            numpy.ndarray: An array of target labels if the input is valid.

        Raises:
            ValueError: If the target is invalid or its length doesn't match max_len.
        """
        # check target
        if target == "auto":
            d_ = core_diameter_m * np.ones(max_len)
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
        target_ = self._check_target(target, self.args.core_diameter_m, max_len)

        # pre-compute indexes
        heat_capacity_jkgk, outer_diameter_m, core_diameter_m, ix = (
            self.morgan_coefficients
        )
        a, b = _profile_bim_avg_coeffs(0.5 * core_diameter_m, 0.5 * outer_diameter_m)

        js = np.nonzero(target_ == Solver_.Names.surf)[0]
        ja = np.nonzero(target_ == Solver_.Names.avg)[0]
        jc = np.nonzero(target_ == Solver_.Names.core)[0]
        jx = np.intersect1d(ix, ja)

        # get correct input for quasi-newton solver
        def newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.current_a = i
            self.joule_heating.__init__(**self.args.__dict__)
            surface_temperature_c = np.ones_like(tg) * np.nan
            core_temperature_c = np.ones_like(tg) * np.nan

            surface_temperature_c[js] = Tmax[js]
            core_temperature_c[js] = tg[js]

            surface_temperature_c[ja] = tg[ja]
            core_temperature_c[ja] = 2 * Tmax[ja] - surface_temperature_c[ja]
            core_temperature_c[jx] = (
                b[jx] * Tmax[jx] - a[jx] * surface_temperature_c[jx]
            ) / (b[jx] - a[jx])

            core_temperature_c[jc] = Tmax[jc]
            surface_temperature_c[jc] = tg[jc]

            return surface_temperature_c, core_temperature_c

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

        Args:
            T (float | numpy.ndarray): Initial temperature profile. Default is an empty numpy array.
            target (str | list[str]): Target specification for the solver. Default is "auto".
            tol (float): Tolerance for the solver. Default is DP.tol.
            maxiter (int): Maximum number of iterations for the solver. Default is DP.maxiter.
            return_err (bool): If True, return the error in the output DataFrame. Default is False.
            return_temp (bool): If True, return the temperature profiles in the output DataFrame. Default is True.
            return_power (bool): If True, return the power profiles in the output DataFrame. Default is True.

        Returns:
            pd.DataFrame: DataFrame containing the steady-state intensity and optionally the error, temperature profiles, and power profiles.
        """

        Tmax, newtheader = self._steady_intensity_header(T, target)

        def balance(i: floatArray, tg: floatArray) -> floatArrayLike:
            surface_temperature_c, core_temperature_c = newtheader(i, tg)
            return self.balance(surface_temperature_c, core_temperature_c)

        def morgan(i: floatArray, tg: floatArray) -> floatArray:
            surface_temperature_c, core_temperature_c = newtheader(i, tg)
            return self.morgan(surface_temperature_c, core_temperature_c)

        # solve system
        s = Solver1T(
            self.args.__dict__,
            type(self.joule_heating),
            type(self.solar_heating),
            type(self.convective_cooling),
            type(self.radiative_cooling),
            type(self.precipitation_cooling),
        )
        r = s.steady_intensity(Tmax, tol=1.0, maxiter=8, return_power=False)
        x, y, cnt, err = quasi_newton_2d(
            balance,
            morgan,
            r[Solver_.Names.current_a].values,
            Tmax,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or cnt == maxiter:
            print(
                f"rstat_analytic max err is {np.max(err):.3E} in {cnt:core_diameter_m} iterations"
            )

        # format output
        df = pd.DataFrame({Solver_.Names.current_a: x})

        if return_err:
            df["err"] = err

        if return_temp or return_power:
            surface_temperature_c, core_temperature_c = newtheader(x, y)
            ambient_temperature_c = self.average(
                surface_temperature_c, core_temperature_c
            )

            if return_temp:
                df[Solver_.Names.tsurf] = surface_temperature_c
                df[Solver_.Names.tavg] = ambient_temperature_c
                df[Solver_.Names.tcore] = core_temperature_c

            if return_power:
                df[Solver_.Names.pjle] = self.joule_heating.value(ambient_temperature_c)
                df[Solver_.Names.psol] = self.solar_heating.value(surface_temperature_c)
                df[Solver_.Names.pcnv] = self.convective_cooling.value(
                    surface_temperature_c
                )
                df[Solver_.Names.prad] = self.radiative_cooling.value(
                    surface_temperature_c
                )
                df[Solver_.Names.ppre] = self.precipitation_cooling.value(
                    surface_temperature_c
                )

        return df
