# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from enum import Enum
from typing import Union, List


class CableLocation(Enum):
    SURFACE = "surf"
    AVERAGE = "avg"
    CORE = "core"


CableLocationListLike = Union[CableLocation, List[CableLocation]]
