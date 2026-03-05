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

from thermohl.solver.solver import Solver
from thermohl.solver.parameters import Parameters
from thermohl.solver.entities import (
    HeatEquationType,
    PowerType as PowerType,
    TemperatureType,
    ModelType,
    VariableType as VariableType,
)
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T
from thermohl.solver.slv3t_legacy import Solver3TL

concreteSolverType = Union[Type[Solver1T], Type[Solver3T], Type[Solver3TL]]


def default_values() -> Dict[str, Any]:
    return Parameters().__dict__


def _factory(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
    model: ModelType = ModelType.IEEE,
) -> Solver:
    solver = create_solver_from_heat_equation(heat_equation)

    model_modules = {
        ModelType.CIGRE: cigrep,
        ModelType.IEEE: ieeep,
        ModelType.OLLA: ollap,
        ModelType.RTE: rtep,
    }

    if model not in model_modules:
        raise ValueError(f"Unsupported solver model: {model}")

    module = model_modules[model]
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
        HeatEquationType.ONE_TEMPERATURE: Solver1T,
        HeatEquationType.THREE_TEMPERATURES: Solver3T,
        HeatEquationType.THREE_TEMPERATURES_LEGACY: Solver3TL,
    }

    if heat_equation not in heat_equations_solvers:
        raise ValueError(f"Invalid HeatEquation value {heat_equation.value}")

    return heat_equations_solvers[heat_equation]


def __solver_model(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
    model: ModelType = ModelType.CIGRE,
) -> Solver:
    """Get a Solver using a given model and heat equation.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.
        model (ModelType): Solver model to use.

    """
    return _factory(dic, heat_equation=heat_equation, model=model)


def cigre(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using CIGRE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic=dic, heat_equation=heat_equation, model=ModelType.CIGRE)


def ieee(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using IEEE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=ModelType.IEEE)


def olla(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using RTE-olla models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=ModelType.OLLA)


def rte(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.ONE_TEMPERATURE,
) -> Solver:
    """Get a Solver using RTE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=ModelType.RTE)
