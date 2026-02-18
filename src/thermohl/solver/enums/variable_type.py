# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from enum import Enum


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
