# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Models to compute equilibrium temperature or max intensity in a conductor."""

from typing import Dict, Any, Optional, Union, Type

from thermohl.power import cigre as cigrep
from thermohl.power import cner as cnerp
from thermohl.power import ieee as ieeep
from thermohl.power import olla as ollap

from thermohl.solver.base import Args, Solver
from thermohl.solver.slv1d import Solver1D
from thermohl.solver.slv1t import Solver1T
from thermohl.solver.slv3t import Solver3T
from thermohl.solver.slv3t_legacy import Solver3TL

concreteSolverType = Union[Type[Solver1T], Type[Solver3T], Type[Solver1D]]


def default_values() -> Dict[str, Any]:
    return Args().__dict__


def _factory(
    dic: Optional[Dict[str, Any]] = None, heateq: str = "1t", model: str = "ieee"
) -> Solver:
    solver: concreteSolverType
    if heateq == "1t":
        solver = Solver1T
    elif heateq == "3t":
        solver = Solver3T
    elif heateq == "3tl":
        solver = Solver3TL
    elif heateq == "1d":
        solver = Solver1D
    else:
        raise ValueError()

    if model == "cigre":
        return solver(
            dic,
            cigrep.JouleHeating,
            cigrep.SolarHeating,
            cigrep.ConvectiveCooling,
            cigrep.RadiativeCooling,
        )
    elif model == "ieee":
        return solver(
            dic,
            ieeep.JouleHeating,
            ieeep.SolarHeating,
            ieeep.ConvectiveCooling,
            ieeep.RadiativeCooling,
        )
    elif model == "olla":
        return solver(
            dic,
            ollap.JouleHeating,
            ollap.SolarHeating,
            ollap.ConvectiveCooling,
            ollap.RadiativeCooling,
        )
    elif model == "cner":
        return solver(
            dic,
            cnerp.JouleHeating,
            cnerp.SolarHeating,
            cnerp.ConvectiveCooling,
            cnerp.RadiativeCooling,
        )
    else:
        raise ValueError()


def cigre(dic: Optional[Dict[str, Any]] = None, heateq: str = "1t") -> Solver:
    """Get a Solver using CIGRE models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, model="cigre")


def ieee(dic: Optional[Dict[str, Any]] = None, heateq: str = "1t") -> Solver:
    """Get a Solver using IEEE models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, model="ieee")


def olla(dic: Optional[Dict[str, Any]] = None, heateq: str = "1t") -> Solver:
    """Get a Solver using RTE-olla models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, model="olla")


def cner(dic: Optional[Dict[str, Any]] = None, heateq: str = "1t") -> Solver:
    """Get a Solver using RTE-cner models.

    Parameters
    ----------
    dic : dict, optional
        Input values. The default is None.
    heateq : str, optional
        Input heat equation.

    """
    return _factory(dic, heateq=heateq, model="cner")
