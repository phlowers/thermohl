# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Union, List

import numpy as np
import numpy.typing as npt
from datetime import datetime, date

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("thermohl")
    except Exception:
        __version__ = "unknown"

floatArrayLike = Union[float, npt.NDArray[np.float64]]
numberArrayLike = Union[float, int, npt.NDArray[np.float64], npt.NDArray[np.int64]]
strListLike = Union[str, List[str]]
dateListLike = Union[date, List[date]]
datetimeListLike = Union[datetime, List[datetime]]

floatArray = npt.NDArray[np.float64]
intArray = npt.NDArray[np.int64]
numberArray = Union[npt.NDArray[np.float64], npt.NDArray[np.int64]]
