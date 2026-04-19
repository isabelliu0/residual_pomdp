"""Viewpoint sampling for information gathering."""

from __future__ import annotations

from collections import deque

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.robots import SingleArmPyBulletRobot
from scipy.spatial.transform import Rotation

from residual_controllers.geometry import invert_pose


def sample_hemisphere_viewpoints(
    center: np.ndarray,
    radius_range: tuple[float, float] = (0.25, 0.45),
    n_samples: int = 50,
    elevation_range: tuple[float, float] = (0.6, 1.3),
    elevation_top_bias: float = 2.0,
    seed: int | None = None,
) -> list[Pose]:
    """Sample viewpoints on a hemisphere looking at center point."""
    rng = np.random.default_rng(seed)

    viewpoints = []
    for _ in range(n_samples):
        azimuth = rng.uniform(0, 2 * np.pi)
        u = rng.uniform(0, 1) ** (1.0 / elevation_top_bias)
        elevation = elevation_range[0] + (elevation_range[1] - elevation_range[0]) * u
        radius = rng.uniform(*radius_range)

        x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation)
        position = (x, y, z)

        look_dir = center - np.array(position)
        look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)

        z_axis = look_dir
        world_up = np.array([0, 0, 1])
        x_axis = np.cross(world_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([0, 1, 0])
        else:
            x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)

        rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
        rotation = Rotation.from_matrix(rot_matrix)
        quat = rotation.as_quat()

        viewpoints.append(Pose(position, tuple(quat)))

    return viewpoints


def filter_reachable_viewpoints(
    viewpoints: list[Pose],
    robot: SingleArmPyBulletRobot,
    camera_to_ee_transform: Pose | None = None,
    object_ids: set[int] | None = None,
    min_clearance: float = 0.05,
    physics_client_id: int = 0,
) -> list[tuple[Pose, tuple[float, ...]]]:
    """Filter viewpoints to those reachable by the robot.

    Args:
        viewpoints: List of desired camera poses.
        robot: Robot to check IK for.
        camera_to_ee_transform: Transform from end-effector to camera.
        object_ids: Object IDs to check clearance against.
        min_clearance: Minimum distance from any object's AABB.
        physics_client_id: PyBullet physics client ID for AABB queries.
    """
    reachable = []

    if camera_to_ee_transform is not None:
        ee_to_camera_inv = invert_pose(camera_to_ee_transform)

    for camera_pose in viewpoints:
        pos = np.array(camera_pose.position)

        if object_ids is not None:
            too_close = False
            for oid in object_ids:
                aabb_min, aabb_max = p.getAABB(oid, physicsClientId=physics_client_id)
                expanded_min = np.array(aabb_min) - min_clearance
                expanded_max = np.array(aabb_max) + min_clearance
                if np.all(pos >= expanded_min) and np.all(pos <= expanded_max):
                    too_close = True
                    break
            if too_close:
                continue

        if camera_to_ee_transform is not None:
            ee_pose = multiply_poses(camera_pose, ee_to_camera_inv)
        else:
            ee_pose = camera_pose

        try:
            joint_positions = inverse_kinematics(
                robot, ee_pose, validate=True, set_joints=False
            )
            reachable.append((camera_pose, tuple(joint_positions)))
        except InverseKinematicsError:
            continue

    return reachable


def compute_region_centroid(
    occupancy_grid,
    z_range: tuple[float, float] = (0.0, 0.1),
) -> np.ndarray | None:
    """Compute centroid of the largest uncertain-region cluster."""
    uncertain_voxels = occupancy_grid.get_unobserved_voxels()
    filtered = [
        (x, y, z) for x, y, z in uncertain_voxels if z_range[0] <= z <= z_range[1]
    ]

    if len(filtered) == 0:
        return None

    voxel_to_xyz = {}
    for xyz in filtered:
        voxel = occupancy_grid._xyz_to_voxel(xyz)  # pylint: disable=protected-access
        voxel_to_xyz[voxel] = xyz

    # NOTE: if the uncertain region is ring-shaped or hollow around the center, this will return the centroid of the largest cluster which may be off-center.   # pylint: disable=line-too-long
    visited: set[tuple[int, int, int]] = set()
    best_centroid = None
    best_size = -1
    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    for voxel in voxel_to_xyz:
        if voxel in visited:
            continue
        queue = deque([voxel])
        visited.add(voxel)
        points = []

        while queue:
            current = queue.popleft()
            points.append(voxel_to_xyz[current])
            for dx, dy, dz in neighbors:
                nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nxt in voxel_to_xyz and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        if len(points) > best_size:
            best_size = len(points)
            best_centroid = np.mean(points, axis=0)

    return best_centroid
