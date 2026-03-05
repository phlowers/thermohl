# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from enum import Enum
from typing import Union, List
import numpy as np


class TargetType(Enum):
    """
    Defines the locations in the cable of the reference temperature
    * SURFACE: the reference temperature is at the surface of the cable
    * AVERAGE: the reference temperature is at an average thickness of the cable
    * CORE: the reference temperature is at the core of the cable
    """

    SURFACE = "surface"
    AVERAGE = "average"
    CORE = "core"


CableLocationListLike = Union[TargetType, List[TargetType]]


class CableType(Enum):
    """
    Defines the type of the cable.
    """

    HOMOGENEOUS = "homogeneous"
    BIMETALLIC = "bimetallic"


CableTypeListLike = Union[CableType, np.ndarray[CableType]]


class HeatEquationType(Enum):
    """
    Defines all the possible heat equation types and their string values.
    * WITH_ONE_TEMPERATURE: computes a single temperature for the cable
    * WITH_THREE_TEMPERATURES: computes three temperatures for the cable (core, surface and average)
    * WITH_THREE_TEMPERATURES_LEGACY: computes three temperatures for the cable (core, surface and average), with specifications
    """

    ONE_TEMPERATURE = "1t"
    THREE_TEMPERATURES = "3t"
    THREE_TEMPERATURES_LEGACY = "3tl"


class PowerType(Enum):
    """
    All the powers involved in the cable heating and cooling.
    * Joule heating : the way the electricity heats the cable (+ magnetic effect for core-metal cables)
    * Solar heating : the way the sun heats the cable
    * Convective cooling : the way the air cools the cable
    * Radiative cooling : the way the cable is cooled down by the radiations it emits
    * Precipitation : the way the rain cools the cable
    """

    JOULE = "joule_power"
    SOLAR = "solar_power"
    CONVECTION = "convection_power"
    RADIATION = "radiation_power"
    RAIN = "precipitation_power"


class ModelType(Enum):
    """
    All the models available in thermohl.
    * cigre : model published by CIGRE
    * ieee : model published by IEEE
    * olla : model developed by RTE's R&D team
    * rte : model developed by RTE
    """

    CIGRE = "cigre"
    IEEE = "ieee"
    OLLA = "olla"
    RTE = "rte"


class TemperatureLocation(Enum):
    """
    Defines all the possible temperature locations for the cable.
    * SURFACE: the temperature at the surface of the cable
    * AVERAGE: the average temperature of the cable
    * CORE: the temperature at the core of the cable
    """

    SURFACE = "surface_temperature"
    AVERAGE = "average_temperature"
    CORE = "core_temperature"


class VariableType(Enum):
    """
    Defines the different types of variables that can be used in a solver.
    * ERROR: used to retrieve the error of the solver
    * TIME: the moment associated with TRANSIT and TEMPERATURE predictions
    * TRANSIT: the electric transit used in the solver
    * TEMPERATURE: the temperature used in the solver
    """

    ERROR = "error"
    TIME = "time"
    TRANSIT = "transit"
    TEMPERATURE = "temperature"
