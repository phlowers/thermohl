# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Base class to build a solver for heat equation."""

from datetime import timedelta
from abc import ABC, abstractmethod
from typing import Type, Any, Optional, Iterable
import numpy.typing as npt
import numpy as np
from numpy import ndarray

from thermohl import (
    floatArrayLike,
    floatArray,
    numberArray,
    numberArrayLike,
    datetimeListLike,
)
from thermohl.power import PowerTerm
from thermohl.solver.parameters import Parameters
from thermohl.solver.entities import PowerType, VariableType


class Solver(ABC):
    """Object to solve a temperature problem.

    The temperature of a conductor is driven by four power terms, two heating
    terms (joule and solar heating) and three cooling terms (convective,
    radiative and precipitation cooling). This class is used to solve a
    temperature problem with the heating and cooling terms passed to its
    __init__ function.
    """

    def __init__(
        self,
        dic: Optional[dict[str, Any]] = None,
        joule: Type[PowerTerm] = PowerTerm,
        solar: Type[PowerTerm] = PowerTerm,
        convective: Type[PowerTerm] = PowerTerm,
        radiative: Type[PowerTerm] = PowerTerm,
        precipitation: Type[PowerTerm] = PowerTerm,
    ) -> None:
        """Create a Solver object.

        Args:
            dic (dict[str, Any] | None): Input values used in power terms. If there is a missing value, a default is used.
            joule (Type[PowerTerm]): Joule heating term class.
            solar (Type[PowerTerm]): Solar heating term class.
            convective (Type[PowerTerm]): Convective cooling term class.
            radiative (Type[PowerTerm]): Radiative cooling term class.
            precipitation (Type[PowerTerm]): Precipitation cooling term class.

        Returns:
            None

        """
        self.args = Parameters(dic)
        self.args.extend()
        self.joule_heating = joule(**self.args.__dict__)
        self.solar_heating = solar(**self.args.__dict__)
        self.convective_cooling = convective(**self.args.__dict__)
        self.radiative_cooling = radiative(**self.args.__dict__)
        self.precipitation_cooling = precipitation(**self.args.__dict__)
        self.args.compress()

    def update(self) -> None:
        self.args.extend()
        self.joule_heating.__init__(**self.args.__dict__)
        self.solar_heating.__init__(**self.args.__dict__)
        self.convective_cooling.__init__(**self.args.__dict__)
        self.radiative_cooling.__init__(**self.args.__dict__)
        self.precipitation_cooling.__init__(**self.args.__dict__)
        self.args.compress()

    def balance(self, conductor_temperature: floatArrayLike) -> floatArrayLike:
        return (
            self.joule_heating.value(conductor_temperature)
            + self.solar_heating.value(conductor_temperature)
            - self.convective_cooling.value(conductor_temperature)
            - self.radiative_cooling.value(conductor_temperature)
            - self.precipitation_cooling.value(conductor_temperature)
        )

    @staticmethod
    def powers() -> tuple[PowerType, PowerType, PowerType, PowerType, PowerType]:
        return (
            PowerType.JOULE,
            PowerType.SOLAR,
            PowerType.CONVECTION,
            PowerType.RADIATION,
            PowerType.RAIN,
        )

    @abstractmethod
    def steady_temperature(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def transient_temperature(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def steady_intensity(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def add_error_and_power_if_needed(
        self, temperature_average, err, output, return_err, return_power
    ):
        self.add_error_if_needed(err, output, return_err)
        self.add_power_if_needed(temperature_average, output, return_power)

    @staticmethod
    def add_error_if_needed(err, output, return_err):
        if return_err:
            output[VariableType.ERROR.value] = err

    def add_power_if_needed(
        self, temperature_average, output, return_power, temperature_surface=None
    ):
        if return_power:
            temperature_surface = (
                temperature_surface
                if temperature_surface is not None
                else temperature_average
            )
            output[PowerType.JOULE.value] = self.joule_heating.value(
                temperature_average
            )
            output[PowerType.SOLAR.value] = self.solar_heating.value(
                temperature_surface
            )
            output[PowerType.CONVECTION.value] = self.convective_cooling.value(
                temperature_surface
            )
            output[PowerType.RADIATION.value] = self.radiative_cooling.value(
                temperature_surface
            )
            output[PowerType.RAIN.value] = self.precipitation_cooling.value(
                temperature_surface
            )

    def _add_input_data_to_result(
        self, result: dict[str, floatArrayLike]
    ) -> dict[str, floatArrayLike]:
        self.args.extend()
        result.update(
            {"input_" + key: value for key, value in self.args.__dict__.items()}
        )
        self.args.compress()
        return result


def reshape(input_array: numberArrayLike, nb_row: int, nb_columns: int) -> numberArray:
    """
    Reshape the input array to the specified dimensions (nr, nc) if possible.

    Args:
        input_array (numberArrayLike): Input array to be reshaped.
        nb_row (int): Desired number of rows for the reshaped array.
        nb_columns (int): Desired number of columns for the reshaped array.

    Returns:
        numberArray: Reshaped array of size (nb_row, nb_columns). If reshaping is not possible,
            returns an array filled with the input_value repeated to fill the dimension (nb_row, nb_columns).

    Raises:
        AttributeError: If the input_array has an invalid shape that cannot be reshaped.
    """
    reshaped_array = ndarray
    try:
        input_shape = input_array.shape
        if len(input_shape) == 1:
            if nb_row == input_shape[0]:
                reshaped_array = np.column_stack(nb_columns * (input_array,))
            elif nb_columns == input_shape[0]:
                reshaped_array = np.vstack(nb_row * (input_array,))
        elif len(input_shape) == 0:
            raise AttributeError()
        else:
            reshaped_array = np.reshape(input_array, (nb_row, nb_columns))
    except AttributeError:
        reshaped_array = input_array * np.ones(
            (nb_row, nb_columns), dtype=type(input_array)
        )
    return reshaped_array


def _set_dates(
    datetime_utc: datetimeListLike,
    offset: floatArray,
    number_of_computations: int,
) -> npt.NDArray[np.datetime64]:
    """
    This function is used in transient temperature computations
    It provides a 2D array of size (len(offset), number_of_computations) such that
    datetime_with_offset[i, j] = datetime_utc[j] + offset[i]

    :param datetime_utc: Datetime or list of datetimes representing the initial times.
    :param offset: Array representing the time offsets in seconds.
    :param number_of_computations: Number of computations.

    :return: 2D array of shape (len(offset), number_of_computations) with datetime values.
    """
    datetime_utc = (
        np.array(datetime_utc)
        if isinstance(datetime_utc, Iterable)
        else np.array(number_of_computations * [datetime_utc])
    )

    number_of_offset = len(offset)
    datetime_with_offset = np.zeros(
        (number_of_offset, number_of_computations), dtype=object
    )
    for i in range(number_of_offset):
        datetime_with_offset[i, :] = datetime_utc + timedelta(seconds=float(offset[i]))

    return datetime_with_offset


def get_time_changing_parameters(args, offset, N, n):
    # get datetime for each offset
    datetime_utc = _set_dates(args.datetime_utc, offset, n)

    # A dict with time-changing quantities (with all elements of size N * n)
    de = {
        "datetime_utc": datetime_utc,
        "transit": reshape(args.transit, N, n),
        "ambient_temperature": reshape(args.ambient_temperature, N, n),
        "wind_azimuth": reshape(args.wind_azimuth, N, n),
        "wind_speed": reshape(args.wind_speed, N, n),
        "ambient_pressure": reshape(args.ambient_pressure, N, n),
        "relative_humidity": reshape(args.relative_humidity, N, n),
        "precipitation_rate": reshape(args.precipitation_rate, N, n),
    }
    del datetime_utc
    return de
