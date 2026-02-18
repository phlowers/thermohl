# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Models to compute equilibrium temperature or max intensity in a conductor."""

from typing import Dict, Any, Optional, Union, Type

from thermohl.power import cigre as cigrep
from thermohl.power import rte as rtep
from thermohl.power import ieee as ieeep
from thermohl.power import olla as ollap

from thermohl.solver.base import Args, Solver
from thermohl.solver.enums.cable_location import CableLocation as CableLocation
from thermohl.solver.enums.heat_equation_type import HeatEquationType
from thermohl.solver.enums.power_type import PowerType as PowerType
from thermohl.solver.enums.temperature_location import (
    TemperatureLocation as TemperatureLocation,
)
from thermohl.solver.enums.solver_type import SolverType
from thermohl.solver.enums.variable_type import VariableType as VariableType
from thermohl.solver.slv1d import Solver1D
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T
from thermohl.solver.slv3t_legacy import Solver3TL

concreteSolverType = Union[
    Type[Solver1T], Type[Solver3T], Type[Solver3TL], Type[Solver1D]
]


def default_values() -> Dict[str, Any]:
    return Args().__dict__


def _factory(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
    model: SolverType = SolverType.SOLVER_IEEE,
) -> Solver:
    solver = create_solver_from_heat_equation(heat_equation)

    solver_modules = {
        SolverType.SOLVER_CIGRE: cigrep,
        SolverType.SOLVER_IEEE: ieeep,
        SolverType.SOLVER_OLLA: ollap,
        SolverType.SOLVER_RTE: rtep,
    }

    if model not in solver_modules:
        raise ValueError(f"Unsupported solver model: {model}")

    module = solver_modules[model]
    return solver(
        dic,
        module.JouleHeating,
        module.SolarHeating,
        module.ConvectiveCooling,
        module.RadiativeCooling,
    )


def create_solver_from_heat_equation(
    heat_equation: HeatEquationType,
) -> concreteSolverType:
    heat_equations_solvers = {
        HeatEquationType.WITH_ONE_TEMPERATURE: Solver1T,
        HeatEquationType.WITH_THREE_TEMPERATURES: Solver3T,
        HeatEquationType.WITH_THREE_TEMPERATURES_LEGACY: Solver3TL,
        HeatEquationType.WITH_1D: Solver1D,
    }

    if heat_equation not in heat_equations_solvers:
        raise ValueError(f"Invalid HeatEquation value {heat_equation.value}")

    return heat_equations_solvers[heat_equation]


def __solver_model(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
    model: SolverType = SolverType.SOLVER_CIGRE,
) -> Solver:
    """Get a Solver using a given model and heat equation.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.
        model (SolverType): Solver model to use.

    """
    return _factory(dic, heat_equation=heat_equation, model=model)


def cigre(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using CIGRE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(
        dic=dic, heat_equation=heat_equation, model=SolverType.SOLVER_CIGRE
    )


def ieee(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using IEEE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(
        dic, heat_equation=heat_equation, model=SolverType.SOLVER_IEEE
    )


def olla(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using RTE-olla models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(
        dic, heat_equation=heat_equation, model=SolverType.SOLVER_OLLA
    )


def rte(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.WITH_ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using RTE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=SolverType.SOLVER_RTE)
