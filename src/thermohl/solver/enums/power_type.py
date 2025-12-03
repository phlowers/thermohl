from enum import Enum

class PowerType(Enum):
    JOULE = "P_joule"
    SOLAR = "P_solar"
    CONVECTION = "P_convection"
    RADIATION = "P_radiation"
    RAIN = "P_precipitation"