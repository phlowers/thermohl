from enum import Enum

class TemperatureLocation(Enum):
    SURFACE = "t_surf"
    AVERAGE = "t_avg"
    CORE = "t_core"