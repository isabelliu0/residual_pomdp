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
    plane_z: float = 0.025,
) -> dict[int, tuple[tuple[float, ...], float]]:
    """Detect objects visible in camera view using segmentation mask.

    Uses ray-plane intersection at plane_z so point_world_z == plane_z
    regardless of camera orientation errors.

    Returns dict[object_id] -> ((x,y,z,qx,qy,qz,qw), confidence)
    """
    detections: dict[int, tuple[tuple[float, ...], float]] = {}

    pos, quat_cam = camera_pose
    ray_origin = np.array(pos, dtype=np.float64)
    rot_matrix = np.array(p.getMatrixFromQuaternion(quat_cam)).reshape(3, 3)
    dz = plane_z - ray_origin[2]

    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays_cam = np.stack(
        [
            (uu - camera_intrinsics.cx) / camera_intrinsics.fx,
            (vv - camera_intrinsics.cy) / camera_intrinsics.fy,
            np.ones((h, w), dtype=np.float64),
        ],
        axis=-1,
    )
    rays_world = rays_cam @ rot_matrix.T  # (H, W, 3)

    for obj_id in object_ids:
        mask = segmentation == obj_id
        if not np.any(mask):
            continue

        d = depth[mask]
        valid = (d > 0) & np.isfinite(d)
        if not np.any(valid):
            continue

        rd = rays_world[mask][valid]
        t = dz / rd[:, 2]
        keep = (np.abs(rd[:, 2]) >= 1e-6) & (t > 0)
        if not np.any(keep):
            continue

        points_world = ray_origin + t[keep, np.newaxis] * rd[keep]
        centroid = points_world.mean(axis=0)

        _, quat_obj = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=physics_client_id
        )
        confidence = float(mask.sum()) / float(mask.size)

        detected_pose = (
            float(centroid[0]),
            float(centroid[1]),
            float(centroid[2]),
            float(quat_obj[0]),
            float(quat_obj[1]),
            float(quat_obj[2]),
            float(quat_obj[3]),
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
