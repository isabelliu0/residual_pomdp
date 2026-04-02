"""Perception utilities for object detection from camera images."""

from __future__ import annotations

import numpy as np
import pybullet as p

from residual_controllers.beliefs.structs import CameraIntrinsics


def detect_objects_from_segmentation(
    segmentation: np.ndarray,
    label_to_id: dict[str, int],
    physics_client_id: int,
    detection_pos_std: float = 0.0,
    detection_distance_ref: float = 0.5,
    camera_pos: tuple[float, ...] | None = None,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Detect objects visible in camera view using segmentation mask.

    Uses ground-truth poses from the simulator for detected objects.
    Segmentation determines visibility; position comes directly from
    pybullet. XY Gaussian noise (detection_pos_std) simulates sensor
    measurement uncertainty.

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

        if detection_pos_std > 0:
            if camera_pos is not None:
                dist = float(np.linalg.norm(np.array(pos_gt) - np.array(camera_pos)))
                effective_std = detection_pos_std * dist / detection_distance_ref
            else:
                effective_std = detection_pos_std
            x = float(pos_gt[0]) + np.random.normal(0, effective_std)
            y = float(pos_gt[1]) + np.random.normal(0, effective_std)
        else:
            x = float(pos_gt[0])
            y = float(pos_gt[1])
        detected_pose = (
            x,
            y,
            float(pos_gt[2]),
            float(quat_obj[0]),
            float(quat_obj[1]),
            float(quat_obj[2]),
            float(quat_obj[3]),
        )
        detections[label] = (detected_pose, confidence)

    return detections


def detect_objects_from_pointcloud_pca(
    segmentation: np.ndarray,
    depth: np.ndarray,
    label_to_id: dict[str, int],
    physics_client_id: int,
    intrinsics: CameraIntrinsics,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    top_face_initial_fraction: float = 0.25,
    top_face_inlier_distance: float = 0.005,
    pixel_noise_std: float = 0.0,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Detect objects via point-cloud PCA on the visible top face.

    For each visible object:
      1. Unproject all segmentation mask pixels to world-frame 3D points.
      2. Top-face selection: take the top top_face_initial_fraction of points
         by world-Z, fit a plane via SVD, keep inliers within
         top_face_inlier_distance of that plane.
      3. XY position: centroid of top-face points.
         Z: top-face plane Z minus half the object AABB height.
      4. Yaw: 2D PCA on top-face XY; roll=pitch=0 (table constraint).

    Returns dict[label] -> ((x,y,z,qx,qy,qz,qw), confidence)
    """
    detections: dict[str, tuple[tuple[float, ...], float]] = {}

    for label, obj_id in label_to_id.items():
        mask = segmentation == obj_id
        if not np.any(mask):
            continue

        ys, xs = np.where(mask)
        points = []
        for y, x in zip(ys, xs):
            d = float(depth[y, x])
            if d <= 0:
                continue
            point_cam = intrinsics.unproject(float(x), float(y), d)
            points.append(transform_point(point_cam, camera_pose))
        if not points:
            continue
        points_world = np.array(points)

        z_threshold = float(
            np.percentile(points_world[:, 2], (1 - top_face_initial_fraction) * 100)
        )
        candidates = points_world[points_world[:, 2] >= z_threshold]
        if len(candidates) < 3:
            continue
        plane_centroid = candidates.mean(axis=0)
        _, _, vh = np.linalg.svd(candidates - plane_centroid, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        distances = np.abs((points_world - plane_centroid) @ normal)
        top_face = points_world[distances <= top_face_inlier_distance]
        if len(top_face) < 3:
            continue

        centroid_xy = top_face[:, :2].mean(axis=0)
        z_center = float(
            p.getBasePositionAndOrientation(obj_id, physicsClientId=physics_client_id)[
                0
            ][2]
        )

        centered = top_face[:, :2] - centroid_xy
        cov = (centered.T @ centered) / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        yaw = float(np.arctan2(principal[1], principal[0]))

        # NOTE: Shift the centroid in pixel space by a single correlated draw
        # scaled by 1/sqrt(N_top_face_pixels): pixel_noise_std is the per-pixel
        # uncertainty, and averaging N independent measurements reduces the
        # centroid std by sqrt(N).  Smaller objects (fewer visible pixels)
        # therefore get proportionally noisier centroid estimates.
        det_x, det_y = float(centroid_xy[0]), float(centroid_xy[1])
        if pixel_noise_std > 0:
            effective_std = pixel_noise_std / float(np.sqrt(max(1, len(top_face))))
            cam_pos_arr = np.array(camera_pose[0])
            rot = np.array(p.getMatrixFromQuaternion(camera_pose[1])).reshape(3, 3)
            point_world_c = np.array([det_x, det_y, float(top_face[:, 2].mean())])
            point_cam_c = rot.T @ (point_world_c - cam_pos_arr)
            xc, yc, zc = point_cam_c
            if zc > 0:
                u_c = (
                    intrinsics.fx * xc / zc
                    + intrinsics.cx
                    + np.random.normal(0.0, effective_std)
                )
                v_c = (
                    intrinsics.fy * yc / zc
                    + intrinsics.cy
                    + np.random.normal(0.0, effective_std)
                )
                xc_n = (u_c - intrinsics.cx) * zc / intrinsics.fx
                yc_n = (v_c - intrinsics.cy) * zc / intrinsics.fy
                p_world_n = rot @ np.array([xc_n, yc_n, zc]) + cam_pos_arr
                det_x, det_y = float(p_world_n[0]), float(p_world_n[1])

        confidence = float(mask.sum()) / float(mask.size)
        detected_pose = (
            det_x,
            det_y,
            z_center,
            0.0,
            0.0,
            float(np.sin(yaw / 2)),
            float(np.cos(yaw / 2)),
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
