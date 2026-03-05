# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from enum import Enum
from typing import Union, List
import numpy as np


class CableLocation(Enum):
    """
    Defines the locations in the cable where the measures are computed.
    * SURFACE: the measure is at the surface of the cable
    * AVERAGE: the average measure of the cable
    * CORE: the measure is at the core of the cable
    """

    SURFACE = "surf"
    AVERAGE = "avg"
    CORE = "core"


CableLocationListLike = Union[CableLocation, List[CableLocation]]


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
    * WITH_1D: computes the temperature with 1D solver
    """

    WITH_ONE_TEMPERATURE = "1t"
    WITH_THREE_TEMPERATURES = "3t"
    WITH_THREE_TEMPERATURES_LEGACY = "3tl"
    WITH_1D = "1d"


class PowerType(Enum):
    """
    All the powers involved in the cable heating and cooling.
    * Joule heating : the way the electricity heats the cable (+ magnetic effect for core-metal cables)
    * Solar heating : the way the sun heats the cable
    * Convective cooling : the way the air cools the cable
    * Radiative cooling : the way the cable is cooled down by the radiations it emits
    * Precipitation : the way the rain cools the cable
    """

    JOULE = "P_joule"
    SOLAR = "P_solar"
    CONVECTION = "P_convection"
    RADIATION = "P_radiation"
    RAIN = "P_precipitation"


class SolverType(Enum):
    """
    All the solvers available in thermohl.
    * cigre : Solver published by CIGRE
    * ieee : Solver published by IEEE
    * olla : Solver developed by RTE's R&D team
    * rte : Solver developed by RTE
    """

    SOLVER_CIGRE = "cigre"
    SOLVER_IEEE = "ieee"
    SOLVER_OLLA = "olla"
    SOLVER_RTE = "rte"


class TemperatureLocation(Enum):
    """
    Defines all the possible temperature locations for the cable.
    * SURFACE: the temperature at the surface of the cable
    * AVERAGE: the average temperature of the cable
    * CORE: the temperature at the core of the cable
    """

    SURFACE = "t_surf"
    AVERAGE = "t_avg"
    CORE = "t_core"


class VariableType(Enum):
    """
    Defines the different types of variables that can be used in a solver.
    * ERROR: used to retrieve the error of the solver
    * TIME: the moment associated with TRANSIT and TEMPERATURE predictions
    * TRANSIT: the electric transit used in the solver
    * TEMPERATURE: the temperature used in the solver
    """

    ERROR = "err"
    TIME = "time"
    TRANSIT = "transit"
    TEMPERATURE = "t"
