# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from enum import Enum


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
