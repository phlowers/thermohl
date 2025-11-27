from enum import Enum

class SolverType(Enum):
    SOLVER_CIGRE = "cigre",
    SOLVER_IEEE = "ieee",
    SOLVER_OLLA = "olla",
    SOLVER_RTE = "rte"