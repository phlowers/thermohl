# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numbers
from typing import Optional

import numpy as np

from thermohl import floatArrayLike, floatArray
from thermohl.solver.solver import Solver as Solver_, get_time_changing_parameters
from thermohl.solver.parameters import DEFAULT_PARAMETERS as default
from thermohl.solver.entities import PowerType, VariableType
from thermohl.utils import bisect_v


class Solver1T(Solver_):
    def steady_temperature(
        self,
        Tmin: float = default.tmin,
        Tmax: float = default.tmax,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> dict[str, np.array]:
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
            dict[str, np.array]: A dictionary with temperature and other results (depending on inputs) in the keys,
            along with input data.

        """

        # solve with bisection
        conductor_temperature, err = bisect_v(
            lambda x: -self.balance(x),
            Tmin,
            Tmax,
            (self.args.get_number_of_computations(),),
            tol,
            maxiter,
        )

        # format output
        result = {
            VariableType.TEMPERATURE.value: conductor_temperature,
        }
        self.add_error_and_power_if_needed(
            conductor_temperature, err, result, return_err, return_power
        )
        result = self._add_input_data_to_result(result)
        return result

    def transient_temperature(
        self,
        offset: floatArray = np.array([]),
        T0: Optional[float] = None,
        return_power: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Compute transient-state temperature.

        Args:
            offset (numpy.ndarray): A 1D array with times (in seconds) when the temperature needs to be computed. The array must contain increasing values (undefined behaviour otherwise).
            T0 (float | None): Initial temperature. If None, the ambient temperature from the internal dict will be used. The default is None.
            return_power (bool, optional): Return power term values. The default is False.

        Returns:
            dict[str, np.ndarray]: A dictionary with temperature and other results (depending on inputs) in the keys, along with input data.
        """

        # get sizes
        n = self.args.get_number_of_computations()
        N = len(offset)
        if N < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get initial temperature
        if T0 is None:
            T0 = (
                self.args.ambient_temperature
                if isinstance(self.args.ambient_temperature, numbers.Number)
                else self.args.ambient_temperature[0]
            )
        time_changing_parameters = get_time_changing_parameters(self.args, offset, N, n)
        # inverse of m*C : shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        conductor_temperature = np.zeros((N, n))
        conductor_temperature[0, :] = T0

        # main time loop
        for i in range(1, N):
            for k, v in time_changing_parameters.items():
                self.args[k] = v[i, :]
            self.update()
            conductor_temperature[i, :] = (
                conductor_temperature[i - 1, :]
                + (offset[i] - offset[i - 1])
                * self.balance(conductor_temperature[i - 1, :])
                * imc
            )

        # save results
        result = {
            VariableType.TIME.value: offset,
            VariableType.TEMPERATURE.value: conductor_temperature,
        }

        # manage return dict 2: powers
        if return_power:
            for power in Solver_.powers():
                result[power.value] = np.zeros_like(conductor_temperature)
            for i in range(N):
                for key in time_changing_parameters.keys():
                    self.args[key] = time_changing_parameters[key][i, :]
                self.update()
                result[PowerType.JOULE.value][i, :] = self.joule_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.SOLAR.value][i, :] = self.solar_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.CONVECTION.value][i, :] = (
                    self.convective_cooling.value(conductor_temperature[i, :])
                )
                result[PowerType.RADIATION.value][i, :] = self.radiative_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RAIN.value][i, :] = self.precipitation_cooling.value(
                    conductor_temperature[i, :]
                )

        # squeeze return values if n is 1
        if n == 1:
            keys = list(result.keys())
            keys.remove(VariableType.TIME.value)
            for key in keys:
                result[key] = result[key][:, 0]

        result = self._add_input_data_to_result(result)

        return result

    def steady_intensity(
        self,
        max_conductor_temperature: floatArrayLike = np.array([]),
        Imin: float = default.imin,
        Imax: float = default.imax,
        tol: float = default.tol,
        maxiter: int = default.maxiter,
        return_err: bool = False,
        return_power: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute steady-state max intensity.

        Compute the maximum intensity that can be run in a conductor without
        exceeding the temperature given in argument.

        Args:
            max_conductor_temperature (float | numpy.ndarray): Maximum temperature.
            Imin (float, optional): Lower bound for intensity. The default is 0.
            Imax (float, optional): Upper bound for intensity. The default is 9999.
            tol (float, optional): Tolerance for temperature error. The default is 1.0E-06.
            maxiter (int, optional): Max number of iterations. The default is 64.
            return_err (bool, optional): Return final error on intensity to check convergence. The default is False.
            return_power (bool, optional): Return power term values. The default is True.

        Returns:
            dict[str, np.ndarray]: A dictionary with maximum intensity and other results (depending on inputs) in the keys,
            along with input data.

        """

        # save transit in arg
        transit = self.args.transit

        # solve with bisection
        shape = (self.args.get_number_of_computations(),)
        T_ = max_conductor_temperature * np.ones(shape)
        joule_heating = (
            self.convective_cooling.value(T_)
            + self.radiative_cooling.value(T_)
            + self.precipitation_cooling.value(T_)
            - self.solar_heating.value(T_)
        )

        def fun(i: floatArray) -> floatArrayLike:
            self.args.transit = i
            self.joule_heating.__init__(**self.args.__dict__)
            return self.joule_heating.value(T_) - joule_heating

        A, err = bisect_v(fun, Imin, Imax, shape, tol, maxiter)

        # restore previous transit
        self.args.transit = transit

        # format output
        result = {VariableType.TRANSIT.value: A}

        self.add_error_and_power_if_needed(
            max_conductor_temperature,
            err,
            result,
            return_err,
            return_power,
        )

        result = self._add_input_data_to_result(result)

        return result
