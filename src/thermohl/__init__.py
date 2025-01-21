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
