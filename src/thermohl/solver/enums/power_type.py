# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from enum import Enum


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
