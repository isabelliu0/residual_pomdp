"""Perception utilities for object detection from camera images."""

from __future__ import annotations

import numpy as np
import pybullet as p

from residual_controllers.beliefs.structs import CameraIntrinsics


def detect_objects_from_segmentation(
    _rgb: np.ndarray,
    depth: np.ndarray,
    segmentation: np.ndarray,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    camera_intrinsics: CameraIntrinsics,
    object_ids: list[int],
    physics_client_id: int,
) -> dict[int, tuple[tuple[float, ...], float]]:
    """Detect objects visible in camera view using segmentation mask.

    Returns dict[object_id] -> ((x,y,z,qx,qy,qz,qw), confidence)
    """
    detections: dict[int, tuple[tuple[float, ...], float]] = {}

    for obj_id in object_ids:
        mask = segmentation == obj_id

        if not np.any(mask):
            continue

        points_3d_world = []
        rows, cols = np.where(mask)

        for row, col in zip(rows, cols):
            d = depth[row, col]
            if d <= 0 or not np.isfinite(d):
                continue

            point_cam = camera_intrinsics.unproject(float(col), float(row), d)
            point_world = transform_point(point_cam, camera_pose)
            points_3d_world.append(point_world)

        if not points_3d_world:
            continue

        centroid = np.mean(points_3d_world, axis=0)

        _, quat = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=physics_client_id
        )

        confidence = float(mask.sum()) / float(mask.size)

        detected_pose = (
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        )
        detections[obj_id] = (detected_pose, confidence)

    return detections


def transform_point(
    point_cam: tuple[float, float, float],
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
) -> tuple[float, float, float]:
    """Transform point from camera frame to world frame using PyBullet."""
    pos, quat = camera_pose

    rot_matrix = p.getMatrixFromQuaternion(quat)
    rot_matrix = np.array(rot_matrix).reshape((3, 3))

    point_cam_arr = np.array(point_cam)
    point_world = rot_matrix @ point_cam_arr + np.array(pos)

    return (float(point_world[0]), float(point_world[1]), float(point_world[2]))
