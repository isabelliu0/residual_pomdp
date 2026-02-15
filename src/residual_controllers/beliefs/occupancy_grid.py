"""3D occupancy grid with log-odds representation."""

from __future__ import annotations

import numpy as np
import pybullet as p
from numba import njit

from residual_controllers.beliefs.structs import CameraIntrinsics


@njit
def _update_ray_numba(
    grid: np.ndarray,
    origin: np.ndarray,
    endpoint: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    update_magnitude: float,
    log_odds_clamp: float,
) -> None:
    """Update voxels along a ray using DDA traversal (Numba-compiled)."""
    direction = endpoint - origin
    length = np.sqrt(np.sum(direction * direction))

    if length < 1e-6:
        return

    direction = direction / length
    num_steps = min(int(length / (resolution * 0.5)), 1000)
    grid_shape = grid.shape

    for i in range(num_steps):
        t = (i / num_steps) * length
        point = origin + t * direction

        ix = int((point[0] - bounds[0, 0]) / resolution)
        iy = int((point[1] - bounds[1, 0]) / resolution)
        iz = int((point[2] - bounds[2, 0]) / resolution)

        if (
            0 <= ix < grid_shape[0]
            and 0 <= iy < grid_shape[1]
            and 0 <= iz < grid_shape[2]
        ):
            grid[ix, iy, iz] -= update_magnitude
            if grid[ix, iy, iz] < -log_odds_clamp:
                grid[ix, iy, iz] = -log_odds_clamp
            elif grid[ix, iy, iz] > log_odds_clamp:
                grid[ix, iy, iz] = log_odds_clamp

    ex = int((endpoint[0] - bounds[0, 0]) / resolution)
    ey = int((endpoint[1] - bounds[1, 0]) / resolution)
    ez = int((endpoint[2] - bounds[2, 0]) / resolution)

    if 0 <= ex < grid_shape[0] and 0 <= ey < grid_shape[1] and 0 <= ez < grid_shape[2]:
        grid[ex, ey, ez] += update_magnitude
        if grid[ex, ey, ez] < -log_odds_clamp:
            grid[ex, ey, ez] = -log_odds_clamp
        elif grid[ex, ey, ez] > log_odds_clamp:
            grid[ex, ey, ez] = log_odds_clamp


@njit
def _unproject_and_update_depth_numba(
    grid: np.ndarray,
    cam_origin: np.ndarray,
    rot_matrix: np.ndarray,
    depth_image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    bounds: np.ndarray,
    resolution: float,
    update_magnitude: float,
    log_odds_clamp: float,
    stride: int,
) -> None:
    """Unproject depth pixels and update grid (Numba-compiled).

    This replaces the outer Python loops with compiled code.
    """
    height, width = depth_image.shape

    for v in range(0, height, stride):
        for u in range(0, width, stride):
            d = depth_image[v, u]

            if d <= 0 or not np.isfinite(d):
                continue

            x_cam = (u - cx) * d / fx
            y_cam = (v - cy) * d / fy
            z_cam = d

            point_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float32)

            point_world = rot_matrix @ point_cam + cam_origin

            _update_ray_numba(
                grid,
                cam_origin,
                point_world,
                bounds,
                resolution,
                update_magnitude,
                log_odds_clamp,
            )


class LogOddsOccupancyGrid:
    """3D voxel grid with log-odds occupancy for static scenes.

    Uses standard log-odds accumulation (no temporal decay). Uncertainty
    grows naturally through ego-pose confidence weighting.
    """

    def __init__(
        self,
        bounds: list[list[float]],
        resolution: float = 0.015,
        log_odds_clamp: float = 10.0,
    ):
        """Initialize occupancy grid.

        Args:
            bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            resolution: Voxel size in meters
            log_odds_clamp: Maximum absolute log-odds value
        """
        self.bounds = np.array(bounds, dtype=np.float32)
        self.resolution = resolution
        self.log_odds_clamp = log_odds_clamp

        self.grid_shape = self._compute_shape()
        self.grid = np.zeros(self.grid_shape, dtype=np.float32)

    def _compute_shape(self) -> tuple[int, int, int]:
        """Compute grid dimensions from bounds and resolution."""
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        shape = tuple((ranges / self.resolution).astype(int))
        return shape  # type: ignore

    def _xyz_to_voxel(self, xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        """Convert world coordinates to voxel indices."""
        x, y, z = xyz
        ix = int((x - self.bounds[0, 0]) / self.resolution)
        iy = int((y - self.bounds[1, 0]) / self.resolution)
        iz = int((z - self.bounds[2, 0]) / self.resolution)
        return (ix, iy, iz)

    def _voxel_to_xyz(self, voxel: tuple[int, int, int]) -> tuple[float, float, float]:
        """Convert voxel indices to world coordinates (voxel center)."""
        ix, iy, iz = voxel
        x = self.bounds[0, 0] + (ix + 0.5) * self.resolution
        y = self.bounds[1, 0] + (iy + 0.5) * self.resolution
        z = self.bounds[2, 0] + (iz + 0.5) * self.resolution
        return (x, y, z)

    def _is_valid_voxel(self, voxel: tuple[int, int, int]) -> bool:
        """Check if voxel indices are within grid bounds."""
        ix, iy, iz = voxel
        return (
            0 <= ix < self.grid_shape[0]
            and 0 <= iy < self.grid_shape[1]
            and 0 <= iz < self.grid_shape[2]
        )

    def update_from_depth(
        self,
        camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
        depth_image: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        ego_pose_confidence: float = 1.0,
        stride: int = 4,
    ) -> None:
        """Update occupancy grid from depth image using Numba-accelerated ray-
        casting.

        Args:
            camera_pose: ((x,y,z), (qx,qy,qz,qw)) in world frame
            depth_image: (H, W) depth values in meters
            camera_intrinsics: Camera parameters
            ego_pose_confidence: Weight for updates (handles ego-pose uncertainty)
            stride: Pixel sampling stride for efficiency
        """
        cam_pos, cam_quat = camera_pose
        cam_origin = np.array(cam_pos, dtype=np.float32)

        rot_matrix = p.getMatrixFromQuaternion(cam_quat)
        rot_matrix = np.array(rot_matrix, dtype=np.float32).reshape((3, 3))

        update_magnitude = 0.5 * ego_pose_confidence

        _unproject_and_update_depth_numba(
            self.grid,
            cam_origin,
            rot_matrix,
            depth_image,
            float(camera_intrinsics.fx),
            float(camera_intrinsics.fy),
            float(camera_intrinsics.cx),
            float(camera_intrinsics.cy),
            self.bounds,
            self.resolution,
            update_magnitude,
            self.log_odds_clamp,
            stride,
        )

    def get_occupancy_probabilities(self) -> np.ndarray:
        """Convert log-odds to probabilities."""
        return 1.0 / (1.0 + np.exp(-self.grid))

    def get_free_voxels(self) -> list[tuple[float, float, float]]:
        """Return xyz coordinates of voxels with occupancy < 0.5."""
        probs = self.get_occupancy_probabilities()
        free_mask = probs < 0.5

        free_coords = []
        for ix, iy, iz in zip(*np.where(free_mask)):
            xyz = self._voxel_to_xyz((ix, iy, iz))
            free_coords.append(xyz)

        return free_coords

    def is_occupied(self, xyz: tuple[float, float, float]) -> bool:
        """Check if position is occupied (prob > 0.5)."""
        voxel = self._xyz_to_voxel(xyz)
        if not self._is_valid_voxel(voxel):
            return False

        prob = 1.0 / (1.0 + np.exp(-self.grid[voxel]))
        return bool(prob > 0.5)

    def get_unobserved_voxels(
        self, prob_threshold: float = 0.4
    ) -> list[tuple[float, float, float]]:
        """Return xyz of voxels with occupancy prob >= threshold."""
        probs = self.get_occupancy_probabilities()
        unobserved_mask = probs >= prob_threshold

        coords = []
        for ix, iy, iz in zip(*np.where(unobserved_mask)):
            xyz = self._voxel_to_xyz((int(ix), int(iy), int(iz)))
            coords.append(xyz)

        return coords

    def visualize(
        self,
        physics_client_id: int,
        prob_threshold: float = 0.4,
        color: tuple[float, float, float] = (0.3, 0.5, 1.0),
        line_width: float = 1.0,
        max_voxels: int = 2000,
        z_range: tuple[float, float] | None = (0.0, 0.06),
    ) -> list[int]:
        """Draw unobserved voxels as wireframe cubes."""
        unobserved = self.get_unobserved_voxels(prob_threshold)

        if z_range is not None:
            unobserved = [
                (x, y, z) for x, y, z in unobserved if z_range[0] <= z <= z_range[1]
            ]

        if len(unobserved) > max_voxels:
            indices = np.random.choice(len(unobserved), max_voxels, replace=False)
            unobserved = [unobserved[i] for i in indices]

        debug_ids = []
        half_size = self.resolution / 2

        for x, y, z in unobserved:
            corners = [
                (x - half_size, y - half_size, z - half_size),
                (x + half_size, y - half_size, z - half_size),
                (x + half_size, y + half_size, z - half_size),
                (x - half_size, y + half_size, z - half_size),
                (x - half_size, y - half_size, z + half_size),
                (x + half_size, y - half_size, z + half_size),
                (x + half_size, y + half_size, z + half_size),
                (x - half_size, y + half_size, z + half_size),
            ]

            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            for i, j in edges:
                debug_id = p.addUserDebugLine(
                    corners[i],
                    corners[j],
                    lineColorRGB=color,
                    lineWidth=line_width,
                    physicsClientId=physics_client_id,
                )
                debug_ids.append(debug_id)

        return debug_ids

    @staticmethod
    def clear_visualization(debug_ids: list[int], physics_client_id: int) -> None:
        """Remove previously drawn debug items."""
        for debug_id in debug_ids:
            p.removeUserDebugItem(debug_id, physicsClientId=physics_client_id)
