"""Next-Best-View planner using ray-casting information gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
from numba import njit
from pybullet_helpers.geometry import Pose
from pybullet_helpers.robots import SingleArmPyBulletRobot

from residual_controllers.beliefs import get_mean_object_position
from residual_controllers.beliefs.occupancy_grid import LogOddsOccupancyGrid
from residual_controllers.beliefs.structs import CameraIntrinsics
from residual_controllers.information_gathering.viewpoint_sampler import (
    compute_region_centroid,
    filter_reachable_viewpoints,
    sample_hemisphere_viewpoints,
)

if TYPE_CHECKING:
    from residual_controllers.beliefs.structs import Belief


@dataclass
class ViewpointCandidate:
    """A candidate viewpoint with its expected information gain."""

    pose: Pose
    joint_config: tuple[float, ...]
    expected_ig: int  # Number of uncertain voxels visible from this viewpoint


def compute_object_target_center(
    belief: Belief,
    object_id: int,
    z_range: tuple[float, float] = (0.0, 0.1),
) -> np.ndarray | None:
    """Compute target center for viewing a specific object.

    For known objects: returns weighted mean position from particles.
    For occluded/unknown objects: returns centroid of uncertain region.
    """
    confidence = belief.object_confidence.get(object_id, 0.0)
    if confidence < 0.1:
        if belief.visibility_grid is not None:
            print("Planning NBV for centroid of uncertain region...")
            return compute_region_centroid(belief.visibility_grid, z_range=z_range)
        return None

    if object_id in belief.known_objects:
        print("Planning NBV for object {object_id}...")
        return get_mean_object_position(belief, object_id)

    if belief.visibility_grid is not None:
        return compute_region_centroid(belief.visibility_grid, z_range=z_range)

    return None


@njit
def _count_uncertain_voxels_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    max_range: float,
    grid: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    free_max_logodds: float,
    occupied_min_logodds: float,
) -> int:
    """Count uncertain voxels along a ray until hitting occupied or max
    range."""
    count = 0
    step_size = resolution * 0.5
    num_steps = int(max_range / step_size)

    for i in range(num_steps):
        t = i * step_size
        point = origin + t * direction

        ix = int((point[0] - bounds[0, 0]) / resolution)
        iy = int((point[1] - bounds[1, 0]) / resolution)
        iz = int((point[2] - bounds[2, 0]) / resolution)

        if not (
            0 <= ix < grid.shape[0]
            and 0 <= iy < grid.shape[1]
            and 0 <= iz < grid.shape[2]
        ):
            break

        log_odds = grid[ix, iy, iz]

        if log_odds > occupied_min_logodds:
            break

        if free_max_logodds <= log_odds <= occupied_min_logodds:
            count += 1

    return count


@njit
def _compute_viewpoint_ig_numba(
    cam_origin: np.ndarray,
    rot_matrix: np.ndarray,
    grid: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    max_range: float,
    free_max_logodds: float,
    occupied_min_logodds: float,
    stride: int,
) -> int:
    """Compute expected IG by ray-casting from viewpoint through occupancy
    grid."""
    total_uncertain = 0

    for v in range(0, height, stride):
        for u in range(0, width, stride):
            x_cam = (u - cx) / fx
            y_cam = (v - cy) / fy
            dir_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float32)
            dir_cam = dir_cam / np.sqrt(np.sum(dir_cam * dir_cam))

            dir_world = rot_matrix @ dir_cam

            count = _count_uncertain_voxels_ray(
                cam_origin,
                dir_world,
                max_range,
                grid,
                bounds,
                resolution,
                free_max_logodds,
                occupied_min_logodds,
            )
            total_uncertain += count

    return total_uncertain


def compute_viewpoint_ig(
    viewpoint: Pose,
    occupancy_grid: LogOddsOccupancyGrid,
    camera_intrinsics: CameraIntrinsics,
    max_range: float = 1.5,
    stride: int = 10,
) -> int:
    """Compute expected information gain for a viewpoint via ray-casting."""
    cam_origin = np.array(viewpoint.position, dtype=np.float32)
    rot_matrix = np.array(
        p.getMatrixFromQuaternion(viewpoint.orientation), dtype=np.float32
    ).reshape((3, 3))

    free_max_logodds = np.log(
        occupancy_grid.thresholds.free_max / (1 - occupancy_grid.thresholds.free_max)
    )
    occupied_min_logodds = np.log(
        occupancy_grid.thresholds.occupied_min
        / (1 - occupancy_grid.thresholds.occupied_min)
    )

    return _compute_viewpoint_ig_numba(
        cam_origin,
        rot_matrix,
        occupancy_grid.grid,
        occupancy_grid.bounds,
        occupancy_grid.resolution,
        float(camera_intrinsics.fx),
        float(camera_intrinsics.fy),
        float(camera_intrinsics.cx),
        float(camera_intrinsics.cy),
        camera_intrinsics.width,
        camera_intrinsics.height,
        max_range,
        float(free_max_logodds),
        float(occupied_min_logodds),
        stride,
    )


class NBVPlanner:
    """Next-Best-View planner for information gathering."""

    def __init__(
        self,
        robot: SingleArmPyBulletRobot,
        camera_intrinsics: CameraIntrinsics,
        camera_to_ee_transform: Pose | None = None,
        n_viewpoint_samples: int = 50,
        radius_range: tuple[float, float] = (0.25, 0.35),
        elevation_range: tuple[float, float] = (0.6, 1.3),
        max_range: float = 1.0,
        ray_stride: int = 10,
    ):
        self.robot = robot
        self.camera_intrinsics = camera_intrinsics
        self.camera_to_ee_transform = camera_to_ee_transform
        self.n_viewpoint_samples = n_viewpoint_samples
        self.radius_range = radius_range
        self.elevation_range = elevation_range
        self.max_range = max_range
        self.ray_stride = ray_stride

    def plan(
        self,
        occupancy_grid: LogOddsOccupancyGrid,
        target_center: np.ndarray | None = None,
        z_range: tuple[float, float] = (0.0, 0.1),
        seed: int | None = None,
    ) -> ViewpointCandidate | None:
        """Plan next-best-view for information gathering.

        Returns the viewpoint with highest expected IG, or None if no
        valid viewpoint.
        """

        if target_center is None:
            target_center = compute_region_centroid(occupancy_grid, z_range=z_range)
            if target_center is None:
                return None

        candidate_poses = sample_hemisphere_viewpoints(
            center=target_center,
            radius_range=self.radius_range,
            n_samples=self.n_viewpoint_samples,
            elevation_range=self.elevation_range,
            seed=seed,
        )

        reachable = filter_reachable_viewpoints(
            candidate_poses, self.robot, self.camera_to_ee_transform
        )
        if len(reachable) == 0:
            return None

        best_candidate = None
        best_ig = -1

        for pose, joint_config in reachable:
            ig = compute_viewpoint_ig(
                pose,
                occupancy_grid,
                self.camera_intrinsics,
                max_range=self.max_range,
                stride=self.ray_stride,
            )

            if ig > best_ig:
                best_ig = ig
                best_candidate = ViewpointCandidate(
                    pose=pose,
                    joint_config=joint_config,
                    expected_ig=ig,
                )

        return best_candidate

    def plan_for_object(
        self,
        belief: Belief,
        object_id: int,
        z_range: tuple[float, float] = (0.0, 0.1),
        seed: int | None = None,
    ) -> ViewpointCandidate | None:
        """Plan next-best-view for observing a specific object.

        For known objects: samples viewpoints around the object's estimated position.
        For occluded/unknown objects: samples viewpoints around uncertain regions.
        """
        if belief.visibility_grid is None:
            return None

        target_center = compute_object_target_center(belief, object_id, z_range=z_range)
        if target_center is None:
            return None

        return self.plan(
            occupancy_grid=belief.visibility_grid,
            target_center=target_center,
            z_range=z_range,
            seed=seed,
        )
