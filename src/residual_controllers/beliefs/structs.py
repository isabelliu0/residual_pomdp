"""Data structures for belief representations."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class BeliefConfig:
    """Configuration for particle filter belief tracking."""

    # Occupancy grid shape
    grid_bounds: list[list[float]] = field(
        default_factory=lambda: [[0.2, 0.8], [-0.4, 0.4], [0.0, 0.15]]
    )
    grid_resolution: float = 0.015

    # Occupancy grid thresholds
    free_max: float = 0.4
    occupied_min: float = 0.6

    # Occupancy grid ray-casting
    grid_stride: int = 10
    free_space_margin: float = 0.015
    free_update: float = 0.3
    occ_update: float = 2.0
    object_thickness: float = 0.05
    dense_stride: int = 2
    dense_window: int = 15
    depth_noise_scale: float = 0.5

    # Particle weight updates
    likelihood_distance_scale: float = 0.1
    missing_pose_likelihood: float = 0.5
    occluded_likelihood: float = 0.95
    visible_fraction_threshold: float = 0.7

    # Object confidence decay
    visible_missing_decay: float = 0.6
    occluded_decay: float = 1.0
    never_seen_decay: float = 0.9
    confidence_known_threshold: float = 0.6

    # Detection noise (simulates sensor measurement uncertainty)
    # Scales linearly with camera-to-object distance; detection_pos_std is the
    # noise std at detection_distance_ref metres.
    detection_pos_std: float = 0.01
    detection_distance_ref: float = 0.5

    # Pose injection from detections
    pose_injection_pos_std: float = 0.01
    pose_injection_rot_std: float = 0.0
    pose_injection_reset_distance: float = 0.1
    pose_injection_min_alpha: float = 0.1

    # Ego-pose confidence
    ego_pose_confidence_threshold: float = 0.05
    ego_pose_boost_scale: float = 0.5


@dataclass(frozen=True)
class TabletopState:
    """Complete state representation for tabletop manipulation.

    Represents robot configuration and object poses. Object poses can be
    None if the object is unknown/occluded.
    """

    joint_positions: tuple[float, ...]  # 9D: 7 arm + 2 fingers
    gripper_open: float  # [0, 1] normalized
    object_poses: dict[str, SE3Pose]  # label ("A","B",...) -> (x,y,z,qx,qy,qz,qw)


@dataclass
class Belief:
    """Particle-based belief state with visibility tracking.

    Follows the visibility-based approach from tampura find_dice
    environment.
    """

    particles: list[TabletopState]
    weights: np.ndarray
    known_objects: set[str]
    occluded_objects: set[str]
    unknown_objects: set[str]
    object_confidence: dict[str, float] = field(default_factory=dict)
    held_object_label: str | None = None
    visibility_grid: LogOddsOccupancyGrid | None = None
