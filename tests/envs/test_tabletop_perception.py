"""Tests for tabletop camera model and depth-based perception."""

from __future__ import annotations

import os

import imageio
import numpy as np
import pybullet as p
from pybullet_helpers.camera import capture_image

from residual_controllers.beliefs.perception import (
    detect_objects_from_pointcloud_pca,
    detect_objects_from_segmentation,
    transform_point,
)
from residual_controllers.beliefs.structs import CameraIntrinsics
from residual_controllers.envs.tabletop_view_occlusion import TabletopViewOcclusionEnv


def _unproject_mask_points(
    segmentation: np.ndarray,
    depth: np.ndarray,
    obj_id: int,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Unproject all segmentation mask pixels for obj_id to world-frame 3D
    points."""
    mask = segmentation == obj_id
    if not np.any(mask):
        return np.empty((0, 3))
    ys, xs = np.where(mask)
    points = []
    for y, x in zip(ys, xs):
        d = float(depth[y, x])
        if d <= 0:
            continue
        point_cam = intrinsics.unproject(float(x), float(y), d)
        point_world = transform_point(point_cam, camera_pose)
        points.append(point_world)
    return np.array(points) if points else np.empty((0, 3))


def _extract_top_face_points(
    points_world: np.ndarray,
    initial_top_fraction: float = 0.25,
    inlier_distance: float = 0.005,
) -> np.ndarray:
    """Find top-face points by fitting a plane to the highest-Z cluster."""
    if len(points_world) < 3:
        return np.empty((0, 3))

    z_threshold = np.percentile(points_world[:, 2], (1 - initial_top_fraction) * 100)
    candidates = points_world[points_world[:, 2] >= z_threshold]

    if len(candidates) < 3:
        return candidates

    centroid = candidates.mean(axis=0)
    _, _, vh = np.linalg.svd(candidates - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal

    distances = np.abs((points_world - centroid) @ normal)
    return points_world[distances <= inlier_distance]


def _pca_yaw(xy_points: np.ndarray) -> float:
    """2D PCA on XY points; return yaw angle of the first principal
    component."""
    centered = xy_points - xy_points.mean(axis=0)
    cov = (centered.T @ centered) / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    return float(np.arctan2(principal[1], principal[0]))


def _min_yaw_error(yaw_est: float, yaw_gt: float, n_fold: int) -> float:
    """Minimum absolute yaw error accounting for n-fold rotational symmetry."""
    period = 2 * np.pi / n_fold
    diff = (yaw_est - yaw_gt) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    diff_sym = diff % period
    if diff_sym > period / 2:
        diff_sym -= period
    return abs(float(diff_sym))


def test_camera_axis_alignment():
    """Camera z-axis is the depth direction: a point 0.3 m along it must
    project to the image centre, and unprojecting its depth at (cx, cy)
    must recover the original world position."""
    env = TabletopViewOcclusionEnv(gui=False, num_objects=1)
    env.reset(seed=42)

    camera_pos, camera_orn = env.get_camera_pose_se3()
    rot = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3, 3)
    camera_forward = rot[:, 2]

    test_depth = 0.3
    sphere_radius = 0.01
    ball_pos = np.array(camera_pos) + test_depth * camera_forward

    pcid = env.physics_client_id
    col_id = p.createCollisionShape(
        p.GEOM_SPHERE, radius=sphere_radius, physicsClientId=pcid
    )
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=sphere_radius,
        rgbaColor=[1, 0, 0, 1],
        physicsClientId=pcid,
    )
    ball_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=tuple(ball_pos),
        physicsClientId=pcid,
    )

    camera_image = env.get_camera_image()
    seg = camera_image.segmentation
    depth = camera_image.depth

    mask = seg == ball_id
    assert np.any(
        mask
    ), f"Ball not visible: ball_pos={ball_pos}, camera_pos={camera_pos}"

    ys, xs = np.where(mask)
    u_centroid = float(np.mean(xs))
    v_centroid = float(np.mean(ys))

    cx, cy = env.camera_intrinsics.cx, env.camera_intrinsics.cy
    assert abs(u_centroid - cx) < 5.0, f"u={u_centroid:.1f} not near cx={cx:.1f}"
    assert abs(v_centroid - cy) < 5.0, f"v={v_centroid:.1f} not near cy={cy:.1f}"

    d_centroid = float(depth[int(round(v_centroid)), int(round(u_centroid))])
    np.testing.assert_allclose(d_centroid, test_depth, atol=sphere_radius + 0.01)

    point_cam = env.camera_intrinsics.unproject(u_centroid, v_centroid, d_centroid)
    point_world = np.array(transform_point(point_cam, (camera_pos, camera_orn)))
    dist = float(np.linalg.norm(point_world - ball_pos))
    assert (
        dist < sphere_radius + 0.01
    ), f"Unprojected point is {dist:.3f} m from ball centre (> {sphere_radius + 0.01})"
    print(
        f"Ground truth ball position: {ball_pos}, unprojected position: {point_world}, distance: {dist:.3f} m"  # pylint: disable=line-too-long
    )

    env.close()


def test_pointcloud_pca_detection_vs_gt():
    """Point-cloud PCA detection finds the same objects as GT, XY error < 5
    cm."""
    env = TabletopViewOcclusionEnv(gui=False, num_objects=5)
    env.reset(seed=7)

    camera_image = env.get_camera_image()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env._get_label_to_id()  # pylint: disable=protected-access
    pcid = env.physics_client_id

    gt_detections = detect_objects_from_segmentation(
        camera_image.segmentation, label_to_id, pcid, detection_pos_std=0.0
    )
    pca_detections = detect_objects_from_pointcloud_pca(
        camera_image.segmentation,
        camera_image.depth,
        label_to_id,
        pcid,
        env.camera_intrinsics,
        camera_pose,
        pixel_noise_std=150.0,
    )

    assert gt_detections, "GT method found nothing"
    assert pca_detections, "PCA method found nothing"
    assert set(pca_detections.keys()) == set(gt_detections.keys())

    for label, obj_id in label_to_id.items():
        if label not in gt_detections:
            continue
        all_points = _unproject_mask_points(
            camera_image.segmentation,
            camera_image.depth,
            obj_id,
            camera_pose,
            env.camera_intrinsics,
        )
        top_face = _extract_top_face_points(all_points)
        effective_std = 150.0 / float(np.sqrt(max(1, len(top_face))))

        gt_pose = gt_detections[label][0]
        pca_pose = pca_detections[label][0]
        gt_xy = np.array(gt_pose[:2])
        pca_xy = np.array(pca_pose[:2])
        xy_error = float(np.linalg.norm(gt_xy - pca_xy))
        gt_yaw = float(p.getEulerFromQuaternion(gt_pose[3:])[2])
        pca_yaw = float(2 * np.arctan2(pca_pose[5], pca_pose[6]))
        yaw_error = _min_yaw_error(pca_yaw, gt_yaw, n_fold=4)
        assert xy_error < 0.05, f"{label}: XY error {xy_error:.3f} m exceeds 5 cm"
        print(
            f"\n{label}: N_top_face={len(top_face)}, effective_std={effective_std:.2f} px"  # pylint: disable=line-too-long
            f"  GT XY={gt_xy}, PCA XY={pca_xy}, XY err={xy_error:.3f} m"
            f"  GT yaw={np.degrees(gt_yaw):.1f}°, PCA yaw={np.degrees(pca_yaw):.1f}°"
            f"  yaw err (4-fold)={np.degrees(yaw_error):.1f}°"
        )

    env.close()


def test_top_face_pca_visualization():
    """Visualize top-face 3D points used for PCA pose estimation."""
    env = TabletopViewOcclusionEnv(gui=False, num_objects=5)
    env.reset(seed=42)

    camera_image = env.get_camera_image()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env._get_label_to_id()  # pylint: disable=protected-access
    pcid = env.physics_client_id

    os.makedirs("videos/perception_test", exist_ok=True)

    print(f"\n{'='*60}")
    print("TOP FACE PCA VISUALIZATION")
    print(f"{'='*60}")

    for label, obj_id in label_to_id.items():
        all_points = _unproject_mask_points(
            camera_image.segmentation,
            camera_image.depth,
            obj_id,
            camera_pose,
            env.camera_intrinsics,
        )
        if len(all_points) == 0:
            print(f"\n{label}: not visible")
            continue

        top_face = _extract_top_face_points(all_points)
        gt_pos, gt_quat = p.getBasePositionAndOrientation(obj_id, physicsClientId=pcid)
        gt_euler = p.getEulerFromQuaternion(gt_quat)

        print(f"\n{label} (id={obj_id}):")
        print(f"  mask points: {len(all_points)}, top-face inliers: {len(top_face)}")
        print(f"  GT pos: ({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f})")
        print(f"  GT yaw: {np.degrees(gt_euler[2]):.1f}\u00b0")

        if len(top_face) >= 3:
            centroid_xy = top_face[:, :2].mean(axis=0)
            yaw_est = _pca_yaw(top_face[:, :2])
            print(f"  centroid XY: ({centroid_xy[0]:.3f}, {centroid_xy[1]:.3f})")
            print(f"  PCA yaw est: {np.degrees(yaw_est):.1f}\u00b0")

            p.addUserDebugPoints(
                top_face.tolist(),
                [[0.0, 0.0, 0.0]] * len(top_face),
                pointSize=5,
                physicsClientId=pcid,
            )

    imageio.imwrite("videos/perception_test/top_face_pca_wrist.png", camera_image.rgb)
    frame = capture_image(
        pcid,
        camera_distance=0.9,
        camera_yaw=50,
        camera_pitch=-35,
        camera_target=(0.5, 0.0, 0.0),
        image_width=640,
        image_height=480,
    )
    imageio.imwrite("videos/perception_test/top_face_pca_external.png", frame)
    print("\nSaved: videos/perception_test/top_face_pca_wrist.png")
    print("Saved: videos/perception_test/top_face_pca_external.png")

    print(f"{'='*60}\n")
    # input("Press Enter to close...")
    env.close()


def test_pca_6d_pose_vs_gt():
    """Compare 6D pose estimated via top-face PCA to ground truth.

    Pipeline: unproject mask pixels -> world-frame point cloud -> plane fit
    to isolate top face -> centroid XY + known Z for position; 2D PCA yaw
    for orientation.
    """
    env = TabletopViewOcclusionEnv(gui=False, num_objects=5)
    env.reset(seed=7)

    camera_image = env.get_camera_image()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env._get_label_to_id()  # pylint: disable=protected-access
    pcid = env.physics_client_id

    HALF_EXTENT = 0.025

    print(f"\n{'='*60}")
    print("PCA 6D POSE ESTIMATE VS GT")
    print(f"{'='*60}")

    found_any = False
    for label, obj_id in label_to_id.items():
        all_points = _unproject_mask_points(
            camera_image.segmentation,
            camera_image.depth,
            obj_id,
            camera_pose,
            env.camera_intrinsics,
        )
        if len(all_points) < 3:
            continue

        top_face = _extract_top_face_points(all_points)
        if len(top_face) < 3:
            continue

        centroid_xy = top_face[:, :2].mean(axis=0)
        z_est = HALF_EXTENT
        yaw_est = _pca_yaw(top_face[:, :2])

        gt_pos, gt_quat = p.getBasePositionAndOrientation(obj_id, physicsClientId=pcid)
        gt_euler = p.getEulerFromQuaternion(gt_quat)
        gt_yaw = gt_euler[2]

        xy_error = float(np.linalg.norm(centroid_xy - np.array(gt_pos[:2])))
        z_error = abs(z_est - float(gt_pos[2]))
        yaw_error_sym = _min_yaw_error(yaw_est, gt_yaw, n_fold=4)

        print(f"\n{label}:")
        print(
            f"  GT  pos: ({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f})"
            f"  yaw={np.degrees(gt_yaw):.1f}\u00b0"
        )
        print(
            f"  est pos: ({centroid_xy[0]:.3f}, {centroid_xy[1]:.3f}, {z_est:.3f})"
            f"  yaw={np.degrees(yaw_est):.1f}\u00b0"
        )
        print(
            f"  XY err: {xy_error*100:.1f}cm  Z err: {z_error*100:.1f}cm"
            f"  yaw err (4-fold sym): {np.degrees(yaw_error_sym):.1f}\u00b0"
        )

        assert xy_error < 0.05, f"{label}: XY error {xy_error:.3f} m > 5 cm"
        assert z_error < 0.005, f"{label}: Z error {z_error:.3f} m > 5 mm"
        found_any = True

    assert found_any, "No visible objects with sufficient top-face points"
    print(f"\n{'='*60}\n")
    env.close()
