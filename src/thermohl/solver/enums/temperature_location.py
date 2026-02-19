# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from enum import Enum


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
