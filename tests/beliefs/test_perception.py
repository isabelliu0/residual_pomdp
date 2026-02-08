"""Tests for perception utilities."""

import numpy as np

from residual_controllers.beliefs import CameraIntrinsics
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


def test_camera_intrinsics_from_pybullet():
    """Test extracting camera intrinsics from PyBullet projection matrix."""
    proj_matrix = [2.0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, -1.0, -1.0, 0, 0, -0.2, 0]
    width, height = 640, 480

    intrinsics = CameraIntrinsics.from_pybullet_projection(proj_matrix, width, height)

    assert intrinsics.width == width
    assert intrinsics.height == height
    assert intrinsics.fx > 0
    assert intrinsics.fy > 0


def test_camera_intrinsics_unproject():
    """Test unprojecting pixel to 3D point."""
    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
    )

    u, v, depth = 320.0, 240.0, 1.0
    point_cam = intrinsics.unproject(u, v, depth)

    assert len(point_cam) == 3
    assert np.isclose(point_cam[2], depth)


def test_object_detection_from_segmentation():
    """Test object detection from segmentation mask."""
    env = TabletopPickEnv(gui=False, num_objects=3)
    _, _ = env.reset(seed=42)

    camera_image = env.get_camera_image()

    assert camera_image.rgb.shape == (480, 640, 3)
    assert camera_image.depth.shape == (480, 640)
    assert camera_image.segmentation.shape == (480, 640)

    assert env.belief is not None
    assert len(env.belief.known_objects) > 0

    env.close()
