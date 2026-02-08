"""Tests for occupancy grid."""

import numpy as np

from residual_controllers.beliefs import CameraIntrinsics, LogOddsOccupancyGrid


def test_occupancy_grid_initialization():
    """Test occupancy grid creation."""
    grid = LogOddsOccupancyGrid(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], resolution=0.1
    )

    assert grid.grid_shape == (10, 10, 10)
    assert grid.grid.shape == (10, 10, 10)


def test_occupancy_grid_update():
    """Test updating occupancy grid from depth image."""
    grid = LogOddsOccupancyGrid(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], resolution=0.1
    )

    depth = np.ones((100, 100)) * 0.5
    camera_pose = ((0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0))
    intrinsics = CameraIntrinsics(
        fx=50.0, fy=50.0, cx=50.0, cy=50.0, width=100, height=100
    )

    grid.update_from_depth(
        camera_pose, depth, intrinsics, ego_pose_confidence=1.0, stride=10
    )

    probs = grid.get_occupancy_probabilities()
    assert probs.shape == (10, 10, 10)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_occupancy_grid_free_voxels():
    """Test getting free voxels from occupancy grid."""
    grid = LogOddsOccupancyGrid(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], resolution=0.1
    )

    free_voxels = grid.get_free_voxels()
    assert isinstance(free_voxels, list)


def test_occupancy_grid_is_occupied():
    """Test checking if position is occupied."""
    grid = LogOddsOccupancyGrid(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], resolution=0.1
    )

    occupied = grid.is_occupied((0.5, 0.5, 0.5))
    assert isinstance(occupied, bool)
