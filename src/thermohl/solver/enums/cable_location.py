from enum import Enum
from typing import Union, List


class CableLocation(Enum):
    SURFACE = "surf"
    AVERAGE = "avg"
    CORE = "core"

CableLocationListLike = Union[CableLocation, List[CableLocation]]
