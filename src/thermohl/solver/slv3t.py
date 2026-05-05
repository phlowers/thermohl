# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Tuple, Type, Optional, Dict, Any, Callable

import numpy as np

from thermohl import floatArrayLike, floatArray, intArray
from thermohl.power import PowerTerm
from thermohl.solver.entities import (
    TargetType,
    CableLocationListLike,
    CableType,
    CableTypeListLike,
    PowerType,
    TemperatureType,
    VariableType,
)
from thermohl.solver.parameters import DEFAULT_PARAMETERS as default
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.solver import (
    Solver as Solver_,
    get_time_changing_parameters,
)
from thermohl.utils import quasi_newton_2d

logger = logging.getLogger(__name__)


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


def _infer_target_from_cable_type(
    cable_type: CableTypeListLike,
    target: CableLocationListLike,
) -> CableLocationListLike:
    """Infer target cable location from cable type: HOMOGENEOUS -> AVERAGE, BIMETALLIC -> CORE.

    If both target and cable_type are provided, target is ignored."""

    if target is not None and cable_type is not None:
        logger.warning(
            "Both target and cable_type are provided. Ignoring given target and using cable_type to determine target instead."
        )

    if cable_type is None:
        return target

    if isinstance(cable_type, CableType):
        if cable_type == CableType.HOMOGENEOUS:
            return TargetType.AVERAGE
        elif cable_type == CableType.BIMETALLIC:
            return TargetType.CORE
    else:
        return [
            TargetType.AVERAGE if ct == CableType.HOMOGENEOUS else TargetType.CORE
            for ct in cable_type
        ]


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
        heat_capacity = 0.5 * np.ones((self.args.get_number_of_computations(),))
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
        self.args.extend()
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
        average_temperature = 0.5 * (surface_temperature + core_temperature)
        _, outer_diameter, core_diameter, positive_surface_diameter_indices = (
            self.morgan_coefficients
        )
        average_temperature[positive_surface_diameter_indices] = _profile_bim_avg(
            surface_temperature[positive_surface_diameter_indices],
            core_temperature[positive_surface_diameter_indices],
            0.5 * core_diameter[positive_surface_diameter_indices],
            0.5 * outer_diameter[positive_surface_diameter_indices],
        )
        return average_temperature

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
        average_temperature = self.average(surface_temperature, core_temperature)
        return self.joule_heating.value(average_temperature)

    def balance(
        self,
        surface_temperature: floatArray,
        core_temperature: floatArray,
        joule_value: Optional[floatArrayLike] = None,
    ) -> floatArrayLike:
        """
        Calculate the thermal balance.

        This method computes the thermal balance by summing the joule heating,
        specific heat, and subtracting the contributions from the cooling
        components (convection, radiation, and conduction).

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.
            joule_value (float | numpy.ndarray, optional): Precomputed joule heating value.
                If None, it will be computed from the given temperatures.

        Returns:
            float | numpy.ndarray: The resulting thermal balance.
        """
        if joule_value is None:
            joule_value = self.joule(surface_temperature, core_temperature)
        return (
            joule_value
            + self.solar_heating.value(surface_temperature)
            - self.convective_cooling.value(surface_temperature)
            - self.radiative_cooling.value(surface_temperature)
            - self.precipitation_cooling.value(surface_temperature)
        )

    def morgan(
        self,
        surface_temperature: floatArray,
        core_temperature: floatArray,
        joule_value: Optional[floatArrayLike] = None,
    ) -> floatArray:
        """
        Computes the Morgan function for given temperature arrays.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.
            joule_value (float | numpy.ndarray, optional): Precomputed joule heating value.
                If None, it will be computed from the given temperatures.

        Returns:
            numpy.ndarray: Resulting array after applying the Morgan function.
        """
        if joule_value is None:
            joule_value = self.joule(surface_temperature, core_temperature)
        heat_capacity = self.morgan_coefficients[0]
        morgan_coefficient = heat_capacity / (
            2.0 * np.pi * self.args.radial_thermal_conductivity
        )
        return (
            core_temperature - surface_temperature
        ) - morgan_coefficient * joule_value

    def balance_and_morgan(
        self, surface_temperature: floatArray, core_temperature: floatArray
    ) -> tuple[floatArrayLike, floatArray]:
        """
        Compute both balance and morgan efficiently by sharing computations.

        This is the optimized version used by steady-state solvers to avoid
        redundant joule heating calculations.

        Args:
            surface_temperature (numpy.ndarray): Array of surface temperatures.
            core_temperature (numpy.ndarray): Array of core temperatures.

        Returns:
            tuple[float | numpy.ndarray, numpy.ndarray]:
                The thermal balance and the Morgan function result.
        """
        # Compute joule once and reuse for both functions
        joule_value = self.joule(surface_temperature, core_temperature)

        balance_value = self.balance(
            surface_temperature, core_temperature, joule_value=joule_value
        )
        morgan_value = self.morgan(
            surface_temperature, core_temperature, joule_value=joule_value
        )
        return balance_value, morgan_value

    def steady_temperature(
        self,
        surface_temperature_guess: Optional[floatArrayLike] = None,
        core_temperature_guess: Optional[floatArrayLike] = None,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> dict[str, np.ndarray]:
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
            dict[str, np.ndarray]: Dictionary containing the steady-state temperatures and optionally the error and power-related values,
            along with input data.
        """

        # if no guess provided, use ambient temp
        shape = (self.args.get_number_of_computations(),)
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
        surface_temperature, core_temperature, iterations, err = quasi_newton_2d(
            self.balance_and_morgan,
            x_init=surface_temperature_guess_,
            y_init=core_temperature_guess_,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or iterations == maxiter:
            logger.debug(
                f"rstat_analytic max err is {np.max(err):.3E} in {iterations:d} iterations"
            )

        # format output
        average_temperature = self.average(surface_temperature, core_temperature)
        result = {
            TemperatureType.SURFACE.value: surface_temperature,
            TemperatureType.AVERAGE.value: average_temperature,
            TemperatureType.CORE.value: core_temperature,
        }

        self.add_error_if_needed(err, result, return_err)
        self.add_power_if_needed(
            average_temperature, result, return_power, surface_temperature
        )

        result = self._add_input_data_to_result(result)

        return result

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
        offset,
        surface_temperature,
        average_temperature,
        core_temperature,
        return_power,
        n,
    ):
        dr = {
            VariableType.TIME.value: offset,
            TemperatureType.SURFACE.value: surface_temperature,
            TemperatureType.AVERAGE.value: average_temperature,
            TemperatureType.CORE.value: core_temperature,
        }

        if return_power:
            for power in Solver_.powers():
                dr[power.value] = np.zeros_like(surface_temperature)

            for i in range(len(offset)):
                dr[PowerType.JOULE.value][i, :] = self.joule(
                    surface_temperature[i, :], core_temperature[i, :]
                )
                dr[PowerType.SOLAR.value][i, :] = self.solar_heating.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.CONVECTION.value][i, :] = self.convective_cooling.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.RADIATION.value][i, :] = self.radiative_cooling.value(
                    surface_temperature[i, :]
                )
                dr[PowerType.RAIN.value][i, :] = self.precipitation_cooling.value(
                    surface_temperature[i, :]
                )

        if n == 1:
            keys = list(dr.keys())
            keys.remove(VariableType.TIME.value)
            for k in keys:
                dr[k] = dr[k][:, 0]

        return dr

    def transient_temperature(
        self,
        offset: floatArray = np.array([]),
        surface_temperature_0: Optional[floatArrayLike] = None,
        core_temperature_0: Optional[floatArrayLike] = None,
        return_power: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute transient-state temperature.

        Args:
            offset (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            surface_temperature_0 (float | numpy.ndarray | None): Initial surface temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            core_temperature_0 (float | numpy.ndarray | None): Initial core temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            Dict[str, Any]: A dictionary with temperature and other results (depending on inputs) in the keys,
            along with input data.

        """
        # get sizes (n for input dict entries, N for time)
        n = self.args.get_number_of_computations()
        N = len(offset)
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
        time_changing_parameters = get_time_changing_parameters(self.args, offset, N, n)
        c1, c2 = self._morgan_transient()
        # inverse of m*C : shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        surface_temperature = np.zeros((N, n))
        average_temperature = np.zeros((N, n))
        core_temperature = np.zeros((N, n))
        surface_temperature[0, :] = surface_temperature_0
        core_temperature[0, :] = core_temperature_0
        average_temperature[0, :] = self.average(
            surface_temperature[0, :], core_temperature[0, :]
        )

        # main time loop
        for i in range(1, len(offset)):
            for k in time_changing_parameters.keys():
                self.args[k] = time_changing_parameters[k][i, :]
            self.update()
            bal = self.balance(
                surface_temperature[i - 1, :], core_temperature[i - 1, :]
            )
            average_temperature[i, :] = (
                average_temperature[i - 1, :] + (offset[i] - offset[i - 1]) * bal * imc
            )
            mrg = c1 * (self.joule_heating.value(average_temperature[i, :]) - bal)
            core_temperature[i, :] = average_temperature[i, :] + c2 * mrg
            surface_temperature[i, :] = core_temperature[i, :] - mrg

        result = self._transient_temperature_results(
            offset,
            surface_temperature,
            average_temperature,
            core_temperature,
            return_power,
            n,
        )
        result = self._add_input_data_to_result(result)
        return result

    @staticmethod
    def _check_target(target: Optional[CableLocationListLike], core_diameter, max_len):
        """
        Validates and processes the target temperature input.

        :param target: The target temperature(s) to be validated. It can be:
            - None: which sets the target automatically.
            - A CableLocation: must be one of CableLocation.SURFACE, CableLocation.AVERAGE, or CableLocation.CORE.
            - A list of CableLocation: each element of the list must be one of CableLocation.SURFACE, CableLocation.AVERAGE, or CableLocation.CORE.
        :param core_diameter: The core diameter of the cable.
        :param max_len: The expected length of the target list if target is a list.
        :return: An array of target labels if the input is valid.
        """
        # check target
        if target is None:
            d_ = core_diameter * np.ones(max_len)
            return np.array(
                [
                    TargetType.CORE if d_[i] > 0.0 else TargetType.AVERAGE
                    for i in range(max_len)
                ]
            )
        elif isinstance(target, TargetType):
            return np.array([target] * max_len)
        return np.array(target)

    def _steady_intensity_header(
        self, T: floatArrayLike, target: CableLocationListLike
    ) -> Tuple[np.ndarray, Callable]:
        """Format input for ampacity solver."""

        max_len = self.args.get_number_of_computations()
        Tmax = T * np.ones(max_len)
        target_ = self._check_target(target, self.args.core_diameter, max_len)

        # pre-compute indexes
        heat_capacity, outer_diameter, core_diameter, ix = self.morgan_coefficients
        a, b = _profile_bim_avg_coeffs(0.5 * core_diameter, 0.5 * outer_diameter)

        js = np.nonzero(target_ == TargetType.SURFACE)[0]
        ja = np.nonzero(target_ == TargetType.AVERAGE)[0]
        jc = np.nonzero(target_ == TargetType.CORE)[0]
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

    def steady_intensity(
        self,
        max_conductor_temperature: floatArrayLike = np.array([]),
        target: CableLocationListLike = None,
        cable_type: CableTypeListLike = None,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_temp: bool = True,
        return_power: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Compute the steady-state intensity for a given temperature profile.

        Args:
            max_conductor_temperature (float | numpy.ndarray): Initial temperature profile. Default is an empty numpy array.
            target (TargetType | list[CableLocation]): Target specification for the solver. Default is None.
            cable_type (CableType | list[CableType]): Cable type specification for the solver. Default is None. If provided, it overrides the target specification.
            tol (float): Tolerance for the solver. Default is DP.tol.
            maxiter (int): Maximum number of iterations for the solver. Default is DP.maxiter.
            return_err (bool): If True, return the error in the output DataFrame. Default is False.
            return_temp (bool): If True, return the temperature profiles in the output DataFrame. Default is True.
            return_power (bool): If True, return the power profiles in the output DataFrame. Default is True.

        Returns:
            dict[str, np.ndarray]: Dictionary containing the steady-state intensity and optionally the error, temperature profiles, and power profiles,
            along with input data.
        """
        target = _infer_target_from_cable_type(cable_type, target)

        Tmax, newtheader = self._steady_intensity_header(
            max_conductor_temperature, target
        )

        def balance_and_morgan(
            i: floatArray, tg: floatArray
        ) -> Tuple[floatArrayLike, floatArray]:
            surface_temperature, core_temperature = newtheader(i, tg)
            return self.balance_and_morgan(surface_temperature, core_temperature)

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
            balance_and_morgan,
            r[VariableType.TRANSIT.value],
            Tmax,
            relative_tolerance=tol,
            max_iterations=maxiter,
            delta_x=1.0e-03,
            delta_y=1.0e-03,
        )
        if np.max(err) > tol or iterations == maxiter:
            logger.debug(
                f"rstat_analytic max err is {np.max(err):.3E} in {iterations:d} iterations"
            )

        # format output
        result = {VariableType.TRANSIT.value: x}

        self.add_error_if_needed(err, result, return_err)

        if return_temp or return_power:
            surface_temperature, core_temperature = newtheader(x, y)
            average_temperature = self.average(surface_temperature, core_temperature)

            if return_temp:
                result[TemperatureType.SURFACE.value] = surface_temperature
                result[TemperatureType.AVERAGE.value] = average_temperature
                result[TemperatureType.CORE.value] = core_temperature

            if return_power:
                result[PowerType.JOULE.value] = self.joule_heating.value(
                    average_temperature
                )
                result[PowerType.SOLAR.value] = self.solar_heating.value(
                    surface_temperature
                )
                result[PowerType.CONVECTION.value] = self.convective_cooling.value(
                    surface_temperature
                )
                result[PowerType.RADIATION.value] = self.radiative_cooling.value(
                    surface_temperature
                )
                result[PowerType.RAIN.value] = self.precipitation_cooling.value(
                    surface_temperature
                )

        result = self._add_input_data_to_result(result)

        return result
