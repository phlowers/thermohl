# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple, Type, Optional, Dict, Any, Callable

import numpy as np
import pandas as pd

from thermohl import floatArrayLike, floatArray, intArray
from thermohl.power import PowerTerm
from thermohl.solver.base import Solver as Solver_, _DEFPARAM as DP, _set_dates, reshape
from thermohl.solver.enums.cable_location import CableLocation, CableLocationListLike
from thermohl.solver.enums.cable_type import CableType, CableTypeListLike
from thermohl.solver.enums.power_type import PowerType
from thermohl.solver.enums.temperature_location import TemperatureLocation
from thermohl.solver.enums.variable_type import VariableType
from thermohl.solver.slv1t import Solver1T
from thermohl.utils import quasi_newton_2d


def _profile_mom(
    surface_temperature: float,
    core_temperature: float,
    radius: floatArrayLike,
    outer_radius: float,
) -> floatArrayLike:
    """Analytic temperature profile for steady heat equation in cylinder (mono-mat)."""
    return surface_temperature + (core_temperature - surface_temperature) * (
        1.0 - (radius / outer_radius) ** 2
    )


def _phi(
    radius: floatArrayLike, core_radius: floatArrayLike, outer_radius: floatArrayLike
) -> floatArrayLike:
    """Primitive function used in _profile_bim*** functions."""
    core_radius_2 = core_radius**2
    return (
        0.5 * (radius**2 - core_radius_2) - core_radius_2 * np.log(radius / core_radius)
    ) / (outer_radius**2 - core_radius_2)


def _profile_bim_avg_coeffs(
    core_radius: floatArrayLike, outer_radius: floatArrayLike
) -> tuple[floatArrayLike, floatArrayLike]:
    core_radius_2 = core_radius**2
    outer_radius_2 = outer_radius**2
    a = (
        0.5 * (outer_radius_2 - core_radius_2) ** 2
        - outer_radius_2
        * core_radius_2
        * (2.0 * np.log(outer_radius / core_radius) - 1.0)
        - core_radius**4
    )
    b = (
        2.0
        * outer_radius_2
        * (outer_radius_2 - core_radius_2)
        * _phi(outer_radius, core_radius, outer_radius)
    )
    return a, b


def _profile_bim_avg(
    surface_temperature: floatArrayLike,
    core_temperature: floatArrayLike,
    core_radius: floatArrayLike,
    outer_radius: floatArrayLike,
) -> floatArrayLike:
    """Analytical formulation for average temperature in _profile_bim."""
    a, b = _profile_bim_avg_coeffs(core_radius, outer_radius)
    return core_temperature - (a / b) * (core_temperature - surface_temperature)


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
            - heat_capacity : numpy.ndarray[float]
                Coefficient array for heat flux.
            - D_ : numpy.ndarray[float]
                Array of core diameters, broadcasted to the shape of `heat_capacity`.
            - d_ : numpy.ndarray[float]
                Array of surface diameters, broadcasted to the shape of `heat_capacity`.
            - positive_surface_diameter_indices : numpy.ndarray[int]
                Indices where surface diameter `d_` is greater than 0.
        """
        heat_capacity = 0.5 * np.ones((self.args.max_len(),))
        outer_diameter = self.args.outer_diameter * np.ones_like(heat_capacity)
        core_diameter = self.args.core_diameter * np.ones_like(heat_capacity)
        positive_surface_diameter_indices = np.nonzero(core_diameter > 0.0)[0]
        heat_capacity[positive_surface_diameter_indices] -= (
            core_diameter[positive_surface_diameter_indices] ** 2
            / (
                outer_diameter[positive_surface_diameter_indices] ** 2
                - core_diameter[positive_surface_diameter_indices] ** 2
            )
        ) * np.log(
            outer_diameter[positive_surface_diameter_indices]
            / core_diameter[positive_surface_diameter_indices]
        )
        return (
            heat_capacity,
            outer_diameter,
            core_diameter,
            positive_surface_diameter_indices,
        )

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
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> floatArrayLike:
        """
        Compute average temperature given surface and core temperature.

        This formula is based on analytical solution in steady-state mode. For
        single material, the formula reduces itself to an usual mean; for
        bi-material conductors, we have geometrical terms to take into account.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: Array of average temperatures.
        """
        ambient_temperature = 0.5 * (surface_temperature + core_temperature)
        _, outer_diameter, core_diameter, positive_surface_diameter_indices = (
            self.morgan_coefficients
        )
        ambient_temperature[positive_surface_diameter_indices] = _profile_bim_avg(
            surface_temperature[positive_surface_diameter_indices],
            core_temperature[positive_surface_diameter_indices],
            0.5 * core_diameter[positive_surface_diameter_indices],
            0.5 * outer_diameter[positive_surface_diameter_indices],
        )
        return ambient_temperature

    def joule(
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> floatArrayLike:
        """
        Calculate the Joule heating effect.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: The calculated Joule heating values.

        Notes:
        - The function computes the average temperature `temperature`.
        - Returns the Joule heating values based on the adjusted temperatures.
        """
        ambient_temperature = self.average(surface_temperature, core_temperature)
        return self.joule_heating.value(ambient_temperature)

    def balance(
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> floatArrayLike:
        """
        Calculate the thermal balance.

        This method computes the thermal balance by summing the joule heating,
        specific heat, and subtracting the contributions from the cooling
        components (convection, radiation, and conduction).

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            float | numpy.ndarray: The resulting thermal balance.
        """
        return (
            self.joule(surface_temperature, core_temperature)
            + self.solar_heating.value(surface_temperature)
            - self.convective_cooling.value(surface_temperature)
            - self.radiative_cooling.value(surface_temperature)
            - self.precipitation_cooling.value(surface_temperature)
        )

    def morgan(
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            numpy.ndarray: Resulting array after applying the Morgan function.
        """
        heat_capacity, _, _, _ = self.morgan_coefficients
        return (core_temperature - surface_temperature) - heat_capacity * self.joule(
            surface_temperature, core_temperature
        ) / (2.0 * np.pi * self.args.radial_thermal_conductivity)

    def steady_temperature(
        self,
        surface_temperature_guess: Optional[floatArrayLike] = None,
        core_temperature_guess: Optional[floatArrayLike] = None,
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the steady-state temperature distribution.

        Args:
            surface_temperature_guess (float | numpy.ndarray | None): Initial guess for the surface temperature. If None, ambient temperature is used.
            core_temperature_guess (float | numpy.ndarray | None): Initial guess for the core temperature. If None, 1.5 times the absolute value of ambient temperature is used.
            tol (float): Tolerance for the quasi-Newton solver.
            maxiter (int): Maximum number of iterations for the quasi-Newton solver.
            return_err (bool): If True, the error of the solution is included in the returned DataFrame.
            return_power (bool): If True, power-related values are included in the returned DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the steady-state temperatures and optionally the error and power-related values.
        """

        # if no guess provided, use ambient temp
        shape = (self.args.max_len(),)
        surface_temperature_guess = (
            surface_temperature_guess
            if surface_temperature_guess is not None
            else 1.0 * self.args.ambient_temperature
        )
        core_temperature_guess = (
            core_temperature_guess
            if core_temperature_guess is not None
            else 1.5 * np.abs(self.args.ambient_temperature)
        )
        surface_temperature_guess_ = surface_temperature_guess * np.ones(shape)
        core_temperature_guess_ = core_temperature_guess * np.ones(shape)

        # solve system
        x, y, iterations, err = quasi_newton_2d(
            func1=self.balance,
            func2=self.morgan,
            x_init=surface_temperature_guess_,
            y_init=core_temperature_guess_,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or iterations == maxiter:
            print(
                f"rstat_analytic max err is {np.max(err):.3E} in {iterations:d} iterations"
            )

        # format output
        z = self.average(x, y)
        df = pd.DataFrame(
            {
                TemperatureLocation.SURFACE: x,
                TemperatureLocation.AVERAGE: z,
                TemperatureLocation.CORE: y,
            }
        )

        if return_err:
            df[VariableType.ERROR] = err

        if return_power:
            df[PowerType.JOULE] = self.joule(x, y)
            df[PowerType.SOLAR] = self.solar_heating.value(x)
            df[PowerType.CONVECTION] = self.convective_cooling.value(x)
            df[PowerType.RADIATION] = self.radiative_cooling.value(x)
            df[PowerType.RAIN] = self.precipitation_cooling.value(x)

        return df

    def _morgan_transient(self):
        """Morgan coefficients for transient temperature."""
        heat_capacity, outer_diameter, core_diameter, ix = self.morgan_coefficients
        c1 = heat_capacity / (2.0 * np.pi * self.args.radial_thermal_conductivity)
        c2 = 0.5 * np.ones_like(c1)
        a, b = _profile_bim_avg_coeffs(
            0.5 * core_diameter[ix], 0.5 * outer_diameter[ix]
        )
        c2[ix] = a / b
        return c1, c2

    def _transient_temperature_results(
        self,
        time,
        surface_temperature,
        ambient_temperature,
        core_temperature,
        return_power,
        n,
    ):
        dr = {
            VariableType.TIME: time,
            TemperatureLocation.SURFACE: surface_temperature,
            TemperatureLocation.AVERAGE: ambient_temperature,
            TemperatureLocation.CORE: core_temperature,
        }

        if return_power:
            for power in Solver_.powers():
                dr[power] = np.zeros_like(surface_temperature)

            for i in range(len(time)):
                dr[PowerType.JOULE][i, :] = self.joule(
                    surface_temperature[i, :], core_temperature[i, :]
                )
                dr[PowerType.SOLAR][i, :] = self.solar_heating.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.CONVECTION][i, :] = self.convective_cooling.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.RADIATION][i, :] = self.radiative_cooling.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.RAIN][i, :] = self.precipitation_cooling.value(
                    surface_temperature[i, :]
                )

        if n == 1:
            keys = list(dr.keys())
            keys.remove(VariableType.TIME)
            for k in keys:
                dr[k] = dr[k][:, 0]

        return dr

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        surface_temperature_0: Optional[floatArrayLike] = None,
        core_temperature_0: Optional[floatArrayLike] = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

        Args:
            time (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            surface_temperature_0 (float | numpy.ndarray | None): Initial surface temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            core_temperature_0 (float | numpy.ndarray | None): Initial core temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
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
        surface_temperature_0 = (
            surface_temperature_0
            if surface_temperature_0 is not None
            else self.args.ambient_temperature
        )
        core_temperature_0 = (
            core_temperature_0
            if core_temperature_0 is not None
            else 1.0 + surface_temperature_0
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
            transit=reshape(self.args.transit, N, n),
            ambient_temperature=reshape(self.args.ambient_temperature, N, n),
            wind_angle=reshape(self.args.wind_angle, N, n),
            wind_speed=reshape(self.args.wind_speed, N, n),
            ambient_pressure=reshape(self.args.ambient_pressure, N, n),
            relative_humidity=reshape(self.args.relative_humidity, N, n),
            precipitation_rate=reshape(self.args.precipitation_rate, N, n),
        )
        del (month, day, hour)

        # shortcuts for time-loop
        c1, c2 = self._morgan_transient()
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        surface_temperature = np.zeros((N, n))
        ambient_temperature = np.zeros((N, n))
        core_temperature = np.zeros((N, n))
        surface_temperature[0, :] = surface_temperature_0
        core_temperature[0, :] = core_temperature_0
        ambient_temperature[0, :] = self.average(
            surface_temperature[0, :], core_temperature[0, :]
        )

        # main time loop
        for i in range(1, len(time)):
            for k in de.keys():
                self.args[k] = de[k][i, :]
            self.update()
            bal = self.balance(
                surface_temperature[i - 1, :], core_temperature[i - 1, :]
            )
            ambient_temperature[i, :] = (
                ambient_temperature[i - 1, :] + (time[i] - time[i - 1]) * bal * imc
            )
            mrg = c1 * (self.joule_heating.value(ambient_temperature[i, :]) - bal)
            core_temperature[i, :] = ambient_temperature[i, :] + c2 * mrg
            surface_temperature[i, :] = core_temperature[i, :] - mrg

        return self._transient_temperature_results(
            time,
            surface_temperature,
            ambient_temperature,
            core_temperature,
            return_power,
            n,
        )

    @staticmethod
    def _check_target(target: CableLocationListLike, core_diameter, max_len):
        """
        Validates and processes the target temperature input.

        Args:
            target (CableLocation | list[CableLocation]): The target temperature(s) to be validated. It can be:
                - None: which sets the target automatically.
                - A CableLocation: must be one of CableLocation.SURFACE, CableLocation.AVERAGE, or CableLocation.CORE.
                - A list of CableLocation: each element of the list must be one of CableLocation.SURFACE, CableLocation.AVERAGE, or CableLocation.CORE.
            max_len (int): The expected length of the target list if target is a list.

        Returns:
            numpy.ndarray: An array of target labels if the input is valid.

        Raises:
            ValueError: If the target is invalid or its length doesn't match max_len.
        """
        # check target
        if target is None:
            d_ = core_diameter * np.ones(max_len)
            target_ = np.array(
                [
                    CableLocation.CORE if d_[i] > 0.0 else CableLocation.AVERAGE
                    for i in range(max_len)
                ]
            )
        elif isinstance(target, CableLocation):
            target_ = np.array([target for _ in range(max_len)])
        else:
            if len(target) != max_len:
                raise ValueError(
                    f"Length of target ({len(target)}) doesn't match max_len {max_len}."
                )
            for t in target:
                if t not in [
                    CableLocation.SURFACE,
                    CableLocation.AVERAGE,
                    CableLocation.CORE,
                ]:
                    raise ValueError()
            target_ = np.array(target)
        return target_

    def _steady_intensity_header(
        self, T: floatArrayLike, target: CableLocationListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.max_len()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.core_diameter, max_len)

        # pre-compute indexes
        heat_capacity, outer_diameter, core_diameter, ix = self.morgan_coefficients
        a, b = _profile_bim_avg_coeffs(0.5 * core_diameter, 0.5 * outer_diameter)

        js = np.nonzero(target_ == CableLocation.SURFACE)[0]
        ja = np.nonzero(target_ == CableLocation.AVERAGE)[0]
        jc = np.nonzero(target_ == CableLocation.CORE)[0]
        jx = np.intersect1d(ix, ja)

        # get correct input for quasi-newton solver
        def newtheader(i: floatArray, tg: floatArray) -> Tuple[floatArray, floatArray]:
            self.args.transit = i
            self.joule_heating.__init__(**self.args.__dict__)
            surface_temperature = np.ones_like(tg) * np.nan
            core_temperature = np.ones_like(tg) * np.nan

            surface_temperature[js] = Tmax[js]
            core_temperature[js] = tg[js]

            surface_temperature[ja] = tg[ja]
            core_temperature[ja] = 2 * Tmax[ja] - surface_temperature[ja]
            core_temperature[jx] = (
                b[jx] * Tmax[jx] - a[jx] * surface_temperature[jx]
            ) / (b[jx] - a[jx])

            core_temperature[jc] = Tmax[jc]
            surface_temperature[jc] = tg[jc]

            return surface_temperature, core_temperature

        return Tmax, newtheader

    def _infer_target_from_cable_type(
        self,
        cable_type: CableTypeListLike,
        target: CableLocationListLike,
    ) -> CableLocationListLike:
        if target is not None and cable_type is not None:
            print(
                "WARNING: Both target and cable_type are provided. Ignoring given target and using cable_type to determine target instead."
            )

        if cable_type is None:
            return target
        else:
            if isinstance(cable_type, CableType):
                if cable_type == CableType.HOMOGENEOUS:
                    return CableLocation.AVERAGE
                elif cable_type == CableType.BIMETALLIC:
                    return CableLocation.CORE
            else:
                return [
                    CableLocation.AVERAGE
                    if ct == CableType.HOMOGENEOUS
                    else CableLocation.CORE
                    for ct in cable_type
                ]

    def steady_intensity(
        self,
        max_conductor_temperature: floatArrayLike = np.array([]),
        target: CableLocationListLike = None,
        cable_type: CableTypeListLike = None,
        tol: float = DP.tol,
        maxiter: int = DP.maxiter,
        return_err: bool = False,
        return_temp: bool = True,
        return_power: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the steady-state intensity for a given temperature profile.

        Args:
            max_conductor_temperature (float | numpy.ndarray): Initial temperature profile. Default is an empty numpy array.
            target (CableLocation | list[CableLocation]): Target specification for the solver. Default is None.
            cable_type (CableType | list[CableType]): Cable type specification for the solver. Default is None. If provided, it overrides the target specification.
            tol (float): Tolerance for the solver. Default is DP.tol.
            maxiter (int): Maximum number of iterations for the solver. Default is DP.maxiter.
            return_err (bool): If True, return the error in the output DataFrame. Default is False.
            return_temp (bool): If True, return the temperature profiles in the output DataFrame. Default is True.
            return_power (bool): If True, return the power profiles in the output DataFrame. Default is True.

        Returns:
            pd.DataFrame: DataFrame containing the steady-state intensity and optionally the error, temperature profiles, and power profiles.
        """
        target = self._infer_target_from_cable_type(cable_type, target)

        Tmax, newtheader = self._steady_intensity_header(
            max_conductor_temperature, target
        )

        def balance(i: floatArray, tg: floatArray) -> floatArrayLike:
            surface_temperature, core_temperature = newtheader(i, tg)
            return self.balance(surface_temperature, core_temperature)

        def morgan(i: floatArray, tg: floatArray) -> floatArray:
            surface_temperature, core_temperature = newtheader(i, tg)
            return self.morgan(surface_temperature, core_temperature)

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
        x, y, iterations, err = quasi_newton_2d(
            balance,
            morgan,
            r[VariableType.TRANSIT].values,
            Tmax,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or iterations == maxiter:
            print(
                f"rstat_analytic max err is {np.max(err):.3E} in {iterations:d} iterations"
            )

        # format output
        df = pd.DataFrame({VariableType.TRANSIT: x})

        if return_err:
            df[VariableType.ERROR] = err

        if return_temp or return_power:
            surface_temperature, core_temperature = newtheader(x, y)
            ambient_temperature = self.average(surface_temperature, core_temperature)

            if return_temp:
                df[TemperatureLocation.SURFACE] = surface_temperature
                df[TemperatureLocation.AVERAGE] = ambient_temperature
                df[TemperatureLocation.CORE] = core_temperature

            if return_power:
                df[PowerType.JOULE] = self.joule_heating.value(ambient_temperature)
                df[PowerType.SOLAR] = self.solar_heating.value(surface_temperature)
                df[PowerType.CONVECTION] = self.convective_cooling.value(
                    surface_temperature
                )
                df[PowerType.RADIATION] = self.radiative_cooling.value(
                    surface_temperature
                )
                df[PowerType.RAIN] = self.precipitation_cooling.value(
                    surface_temperature
                )

        return df
