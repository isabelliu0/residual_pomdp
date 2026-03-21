"""Perception utilities for object detection from camera images."""

from __future__ import annotations

import numpy as np
import pybullet as p


def detect_objects_from_segmentation(
    segmentation: np.ndarray,
    label_to_id: dict[str, int],
    physics_client_id: int,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Detect objects visible in camera view using segmentation mask.

    Uses ground-truth poses from the simulator for detected objects.
    Segmentation determines visibility; position comes directly from
    pybullet.

    Returns dict[label] -> ((x,y,z,qx,qy,qz,qw), confidence)
    """
    detections: dict[str, tuple[tuple[float, ...], float]] = {}

    for label, obj_id in label_to_id.items():
        mask = segmentation == obj_id
        if not np.any(mask):
            continue

        pos_gt, quat_obj = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=physics_client_id
        )
        confidence = float(mask.sum()) / float(mask.size)

        detected_pose = (
            float(pos_gt[0]),
            float(pos_gt[1]),
            float(pos_gt[2]),
            float(quat_obj[0]),
            float(quat_obj[1]),
            float(quat_obj[2]),
            float(quat_obj[3]),
        )
        detections[label] = (detected_pose, confidence)

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
