# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Power terms implementation using CIGRE recommendations.

See Thermal behaviour of overhead conductors, study committee 22, working
group 12, 2002.
"""

from .air import Air
from .solar_heating import SolarHeating
from .convective_cooling import ConvectiveCooling
from .joule_heating import JouleHeating
from .radiative_cooling import RadiativeCooling
