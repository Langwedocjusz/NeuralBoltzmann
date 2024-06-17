"""This module defines simulation config class, which is used by all lbm related code."""

from enum import Enum
from dataclasses import dataclass

class BC(Enum):
    """Enum representing supported types of boundary conditions."""
    PERIODIC = 0
    VON_NEUMANN = 1

@dataclass(slots=True)
class SimulationConfig:
    """Dataclass representing configuration of the lbm simulation."""
    grid_size_x: int
    grid_size_y: int
    tau: float
    gravity: (float, float) = (0.0, 0.0)
    boundary_conditions: (BC, BC, BC, BC) = (
        BC.PERIODIC, BC.PERIODIC, BC.PERIODIC, BC.PERIODIC
    )