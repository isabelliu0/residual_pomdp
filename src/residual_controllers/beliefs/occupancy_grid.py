"""3D occupancy grid with log-odds representation."""

from __future__ import annotations

import numpy as np
import pybullet as p

from residual_controllers.beliefs.structs import CameraIntrinsics


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
        """Update occupancy grid from depth image using ray-casting.

        Args:
            camera_pose: ((x,y,z), (qx,qy,qz,qw)) in world frame
            depth_image: (H, W) depth values in meters
            camera_intrinsics: Camera parameters
            ego_pose_confidence: Weight for updates (handles ego-pose uncertainty)
            stride: Pixel sampling stride for efficiency
        """
        cam_pos, cam_quat = camera_pose
        cam_origin = np.array(cam_pos, dtype=np.float32)

        height, width = depth_image.shape

        rot_matrix = p.getMatrixFromQuaternion(cam_quat)
        rot_matrix = np.array(rot_matrix, dtype=np.float32).reshape((3, 3))

        for v in range(0, height, stride):
            for u in range(0, width, stride):
                d = depth_image[v, u]
                if d <= 0 or not np.isfinite(d):
                    continue

                point_cam = np.array(
                    camera_intrinsics.unproject(float(u), float(v), d), dtype=np.float32
                )
                point_world = rot_matrix @ point_cam + cam_origin

                self._update_ray(
                    cam_origin,
                    point_world,
                    update_magnitude=0.5 * ego_pose_confidence,
                )

    def _update_ray(
        self, origin: np.ndarray, endpoint: np.ndarray, update_magnitude: float
    ) -> None:
        """Update voxels along a ray using DDA traversal."""
        direction = endpoint - origin
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return

        direction = direction / length

        num_steps = int(length / (self.resolution * 0.5))
        num_steps = min(num_steps, 1000)

        for i in range(num_steps):
            t = (i / num_steps) * length
            point = origin + t * direction
            voxel = self._xyz_to_voxel(tuple(point))

            if not self._is_valid_voxel(voxel):
                continue

            self.grid[voxel] -= update_magnitude
            self.grid[voxel] = np.clip(
                self.grid[voxel], -self.log_odds_clamp, self.log_odds_clamp
            )

        endpoint_voxel = self._xyz_to_voxel(tuple(endpoint))
        if self._is_valid_voxel(endpoint_voxel):
            self.grid[endpoint_voxel] += update_magnitude
            self.grid[endpoint_voxel] = np.clip(
                self.grid[endpoint_voxel], -self.log_odds_clamp, self.log_odds_clamp
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
