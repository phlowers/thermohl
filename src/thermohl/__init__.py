from typing import Union
import numpy as np
import numpy.typing as npt

floatArrayLike = Union[float, npt.NDArray[np.float64]]
intArrayLike = Union[int, npt.NDArray[np.int64]]

floatArray = npt.NDArray[np.float64]
intArray = npt.NDArray[np.int64]
