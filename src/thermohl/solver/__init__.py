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
from thermohl.solver.heat_equation_enum import HeatEquationType
from thermohl.solver.slv1d import Solver1D
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T
from thermohl.solver.slv3t_legacy import Solver3TL
from thermohl.solver.solver_type import SolverType

concreteSolverType = Union[Type[Solver1T], Type[Solver3T], Type[Solver3TL], Type[Solver1D]]


def default_values() -> Dict[str, Any]:
    return Args().__dict__


def _factory(
    dic: Optional[Dict[str, Any]] = None,
    heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE,
    model: SolverType = SolverType.SOLVER_CIGRE,
) -> Solver:
    solver: concreteSolverType
    solver = create_solver_from_heat_equation(heat_equation)

    if model == SolverType.SOLVER_CIGRE:
        return solver(
            dic,
            cigrep.JouleHeating,
            cigrep.SolarHeating,
            cigrep.ConvectiveCooling,
            cigrep.RadiativeCooling,
        )
    elif model == SolverType.SOLVER_IEEE:
        return solver(
            dic,
            ieeep.JouleHeating,
            ieeep.SolarHeating,
            ieeep.ConvectiveCooling,
            ieeep.RadiativeCooling,
        )
    elif model == SolverType.SOLVER_OLLA:
        return solver(
            dic,
            ollap.JouleHeating,
            ollap.SolarHeating,
            ollap.ConvectiveCooling,
            ollap.RadiativeCooling,
        )
    elif model == SolverType.SOLVER_RTE:
        return solver(
            dic,
            rtep.JouleHeating,
            rtep.SolarHeating,
            rtep.ConvectiveCooling,
            rtep.RadiativeCooling,
        )
    else:
        raise ValueError()


def create_solver_from_heat_equation(heat_equation: HeatEquationType):
    if heat_equation == HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE:
        solver = Solver1T
    elif heat_equation == HeatEquationType.HEAT_EQUATION_THREE_TEMPERATURES:
        solver = Solver3T
    elif heat_equation == HeatEquationType.HEAT_EQUATION_THREE_TEMPERATURES_LEGACY:
        solver = Solver3TL
    elif heat_equation == HeatEquationType.HEAT_EQUATION_1D:
        solver = Solver1D
    else:
        raise ValueError(f"Invalid HeatEquation value {heat_equation.value}")

    return solver


def __solver_model(dic: Optional[Dict[str, Any]] = None,
                   heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE,
                   model: SolverType = SolverType.SOLVER_CIGRE,
                   )\
        -> Solver:
    """Get a Solver using a given model and heat equation.

       Args:
           dic (dict | None): Input values. The default is None.
           heat_equation (HeatEquationType): Input heat equation.
           model (SolverType): Solver model to use.

       """
    return _factory(dic, heat_equation=heat_equation, model=model)


def cigre(dic: Optional[Dict[str, Any]] = None,
          heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE)\
        -> Solver:
    """Get a Solver using CIGRE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic=dic, heat_equation=heat_equation, model=SolverType.SOLVER_CIGRE)


def ieee(dic: Optional[Dict[str, Any]] = None, heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE) -> Solver:
    """Get a Solver using IEEE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=SolverType.SOLVER_IEEE)


def olla(dic: Optional[Dict[str, Any]] = None, heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE) -> Solver:
    """Get a Solver using RTE-olla models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=SolverType.SOLVER_OLLA)


def rte(dic: Optional[Dict[str, Any]] = None, heat_equation: HeatEquationType = HeatEquationType.HEAT_EQUATION_ONE_TEMPERATURE) -> Solver:
    """Get a Solver using RTE models.

    Args:
        dic (dict | None): Input values. The default is None.
        heat_equation (HeatEquationType): Input heat equation.

    """
    return __solver_model(dic, heat_equation=heat_equation, model=SolverType.SOLVER_RTE)
