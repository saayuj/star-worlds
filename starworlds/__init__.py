"""
Star Worlds — implementation of:
  "Creating Star Worlds: Reshaping the Robot Workspace for Online Motion Planning"
  Dahlin & Karayiannidis, IEEE Trans. Robotics 39(5), 2023.
  https://doi.org/10.1109/TRO.2023.3279029
"""

from .obstacles import Ellipse, ConvexPolygon2D, Polygon2D, Ellipsoid
from .star_world import create_star_world, StarWorldResult, ObstacleProxy
from .integration import StarWorldUpdater

__all__ = [
    "Ellipse",
    "ConvexPolygon2D",
    "Polygon2D",
    "Ellipsoid",
    "create_star_world",
    "StarWorldResult",
    "ObstacleProxy",
    "StarWorldUpdater",
]
