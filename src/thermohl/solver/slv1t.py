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
from thermohl.solver.enums.power_type import PowerType
from thermohl.solver.enums.variable_type import VariableType
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
        conductor_temperature, err = bisect_v(
            lambda x: -self.balance(x), Tmin, Tmax, (self.args.max_len(),), tol, maxiter
        )

        # format output
        df = pd.DataFrame(
            data=conductor_temperature, columns=[VariableType.TEMPERATURE]
        )

        if return_err:
            df[VariableType.ERROR] = err

        if return_power:
            df[PowerType.JOULE] = self.joule_heating.value(conductor_temperature)
            df[PowerType.SOLAR] = self.solar_heating.value(conductor_temperature)
            df[PowerType.CONVECTION] = self.convective_cooling.value(
                conductor_temperature
            )
            df[PowerType.RADIATION] = self.radiative_cooling.value(
                conductor_temperature
            )
            df[PowerType.RAIN] = self.precipitation_cooling.value(conductor_temperature)

        return df

    def transient_temperature(
        self,
        time: floatArray = np.array([]),
        T0: Optional[float] = None,
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

        # get sizes
        args_size = self.args.max_len()
        time_size = len(time)
        if time_size < 2:
            raise ValueError("The length of the time array must be at least 2.")

        # get initial temperature
        if T0 is None:
            T0 = (
                self.args.ambient_temperature
                if isinstance(self.args.ambient_temperature, numbers.Number)
                else self.args.ambient_temperature[0]
            )

        # get month, day and hours
        month, day, hour = _set_dates(
            self.args.month, self.args.day, self.args.hour, time, args_size
        )

        # A dict with time-changing quantities (with all elements of size time_size * args_size)
        de = dict(
            month=month,
            day=day,
            hour=hour,
            transit=reshape(self.args.transit, time_size, args_size),
            ambient_temperature=reshape(
                self.args.ambient_temperature, time_size, args_size
            ),
            wind_angle=reshape(self.args.wind_angle, time_size, args_size),
            wind_speed=reshape(self.args.wind_speed, time_size, args_size),
            ambient_pressure=reshape(self.args.ambient_pressure, time_size, args_size),
            relative_humidity=reshape(
                self.args.relative_humidity, time_size, args_size
            ),
            precipitation_rate=reshape(
                self.args.precipitation_rate, time_size, args_size
            ),
        )
        del (month, day, hour)

        # shortcuts for time-loop
        imc = 1.0 / (self.args.linear_mass * self.args.heat_capacity)

        # init
        conductor_temperature = np.zeros((time_size, args_size))
        conductor_temperature[0, :] = T0

        # main time loop
        for i in range(1, len(time)):
            for k, v in de.items():
                self.args[k] = v[i, :]
            self.update()
            conductor_temperature[i, :] = (
                conductor_temperature[i - 1, :]
                + (time[i] - time[i - 1])
                * self.balance(conductor_temperature[i - 1, :])
                * imc
            )

        # save results
        result = {
            VariableType.TIME: time,
            VariableType.TEMPERATURE: conductor_temperature,
        }

        # manage return dict 2: powers
        if return_power:
            for power in Solver_.powers():
                result[power] = np.zeros_like(conductor_temperature)
            for i in range(time_size):
                for key in de.keys():
                    self.args[key] = de[key][i, :]
                self.update()
                result[PowerType.JOULE][i, :] = self.joule_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.SOLAR][i, :] = self.solar_heating.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.CONVECTION][i, :] = self.convective_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RADIATION][i, :] = self.radiative_cooling.value(
                    conductor_temperature[i, :]
                )
                result[PowerType.RAIN][i, :] = self.precipitation_cooling.value(
                    conductor_temperature[i, :]
                )

        # squeeze return values if args_size is 1
        if args_size == 1:
            keys = list(result.keys())
            keys.remove(VariableType.TIME)
            for key in keys:
                result[key] = result[key][:, 0]

        return result

    def steady_intensity(
        self,
        max_conductor_temperature: floatArrayLike = np.array([]),
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
            max_conductor_temperature (float | numpy.ndarray): Maximum temperature.
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
        transit = self.args.transit

        # solve with bisection
        shape = (self.args.max_len(),)
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
        df = pd.DataFrame(data=A, columns=[VariableType.TRANSIT])

        if return_err:
            df[VariableType.ERROR] = err

        if return_power:
            df[PowerType.JOULE] = self.joule_heating.value(max_conductor_temperature)
            df[PowerType.SOLAR] = self.solar_heating.value(max_conductor_temperature)
            df[PowerType.CONVECTION] = self.convective_cooling.value(
                max_conductor_temperature
            )
            df[PowerType.RADIATION] = self.radiative_cooling.value(
                max_conductor_temperature
            )
            df[PowerType.RAIN] = self.precipitation_cooling.value(
                max_conductor_temperature
            )

        return df
