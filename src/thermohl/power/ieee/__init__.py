# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Power terms implementation using IEEE std 38-2012 models.

IEEE std 38-2012 is the IEEE Standard for Calculating the Current-Temperature
Relationship of Bare Overhead Conductors.
"""

from .air import Air
from .convective_cooling import ConvectiveCooling
from ..convective_cooling import ConvectiveCoolingBase
from .joule_heating import JouleHeating
from .radiative_cooling import RadiativeCooling
from .solar_heating import SolarHeating
