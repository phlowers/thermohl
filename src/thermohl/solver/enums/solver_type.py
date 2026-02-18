# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from enum import Enum


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
