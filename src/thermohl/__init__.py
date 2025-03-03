# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Union, List

import numpy as np
import numpy.typing as npt
import scipy

frozen_dist = scipy.stats._distn_infrastructure.rv_continuous_frozen

floatArrayLike = Union[float, npt.NDArray[np.float64]]
intArrayLike = Union[int, npt.NDArray[np.int64]]
numberArrayLike = Union[float, int, npt.NDArray[np.float64], npt.NDArray[np.int64]]
strListLike = Union[str, List[str]]

floatArray = npt.NDArray[np.float64]
intArray = npt.NDArray[np.int64]
numberArray = Union[npt.NDArray[np.float64], npt.NDArray[np.int64]]
