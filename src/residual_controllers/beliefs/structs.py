"""Data structures for belief representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from residual_controllers.beliefs.occupancy_grid import LogOddsOccupancyGrid

SE3Pose = tuple[float, float, float, float, float, float, float]


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y
    width: int
    height: int

    @classmethod
    def from_pybullet_projection(
        cls, proj_matrix: list[float], width: int, height: int
    ) -> CameraIntrinsics:
        """Extract intrinsics from PyBullet projection matrix.

        PyBullet projection matrix has:
        proj_matrix[0] = 2*fx/width
        proj_matrix[5] = 2*fy/height
        """
        fx = proj_matrix[0] * width / 2
        fy = proj_matrix[5] * height / 2
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx, fy, cx, cy, width, height)

    def unproject(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """Unproject pixel (u, v) at depth to 3D point in camera frame."""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return (x, y, z)


@dataclass(frozen=True)
class TabletopState:
    """Complete state representation for tabletop manipulation.

    Represents robot configuration and object poses. Object poses can be
    None if the object is unknown/occluded.
    """

    joint_positions: tuple[float, ...]  # 9D: 7 arm + 2 fingers
    gripper_open: float  # [0, 1] normalized
    object_poses: dict[int, SE3Pose | None]  # object_id -> (x,y,z,qx,qy,qz,qw) or None


@dataclass
class Belief:
    """Particle-based belief state with visibility tracking.

    Follows the visibility-based approach from tampura find_dice
    environment.
    """

    particles: list[TabletopState]
    weights: np.ndarray  # shape (N,), normalized to sum to 1
    known_objects: set[int]  # Objects currently detected
    occluded_objects: set[int]  # Previously known, now in occluded region
    unknown_objects: set[int]  # Never detected or lost (expected visible but not found)
    held_object_id: int | None = None  # Currently grasped object
    visibility_grid: LogOddsOccupancyGrid | None = None
