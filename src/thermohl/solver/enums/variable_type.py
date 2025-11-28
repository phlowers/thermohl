# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from enum import Enum
from typing import Union, List


class VariableType(Enum):

    POWER_JOULE = "P_joule"
    POWER_SUN = "P_solar"
    POWER_CONVECTION = "P_convection"
    POWER_RADIATION = "P_radiation"
    POWER_RAIN = "P_precipitation"
    ERROR = "err"
    SURFACE = "surf"
    AVERAGE = "avg"
    CORE = "core"
    TIME = "time"
    TRANSIT = "I"
    TEMPERATURE = "t"
    TEMPERATURE_SURFACE = "t_surf"
    TEMPERATURE_AVERAGE = "t_avg"
    TEMPERATURE_CORE = "t_core"


VariableTypeListLike = Union[VariableType, List[VariableType]]