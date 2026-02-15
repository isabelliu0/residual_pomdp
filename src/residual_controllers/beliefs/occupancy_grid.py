"""3D occupancy grid with log-odds representation."""

from __future__ import annotations

import numpy as np
import pybullet as p
from numba import njit

from residual_controllers.beliefs.structs import BeliefConfig, CameraIntrinsics


@njit
def _update_ray_numba(
    grid: np.ndarray,
    origin: np.ndarray,
    endpoint: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    free_update: float,
    occ_update: float,
    log_odds_clamp: float,
    free_space_margin: float,
    is_object_ray: bool,
    object_thickness: float,
    depth_noise_scale: float,
) -> None:
    """Update voxels along a ray with asymmetric updates and object
    thickness."""
    direction = endpoint - origin
    length = np.sqrt(np.sum(direction * direction))

    if length < 1e-6:
        return

    direction = direction / length
    num_steps = min(int(length / (resolution * 0.5)), 1000)
    grid_shape = grid.shape
    free_space_length = length - free_space_margin

    distance_scale = 1.0 / (1.0 + depth_noise_scale * length)
    scaled_free = free_update * distance_scale
    scaled_occ = occ_update * distance_scale

    for i in range(num_steps):
        t = (i / num_steps) * length
        if t >= free_space_length:
            break
        point = origin + t * direction

        ix = int((point[0] - bounds[0, 0]) / resolution)
        iy = int((point[1] - bounds[1, 0]) / resolution)
        iz = int((point[2] - bounds[2, 0]) / resolution)

        if (
            0 <= ix < grid_shape[0]
            and 0 <= iy < grid_shape[1]
            and 0 <= iz < grid_shape[2]
        ):
            grid[ix, iy, iz] -= scaled_free
            if grid[ix, iy, iz] < -log_odds_clamp:
                grid[ix, iy, iz] = -log_odds_clamp

    if is_object_ray:
        thickness_steps = max(1, int(object_thickness / resolution))
        for step in range(thickness_steps):
            occ_point = endpoint + direction * (step * resolution)
            ox = int((occ_point[0] - bounds[0, 0]) / resolution)
            oy = int((occ_point[1] - bounds[1, 0]) / resolution)
            oz = int((occ_point[2] - bounds[2, 0]) / resolution)

            if (
                0 <= ox < grid_shape[0]
                and 0 <= oy < grid_shape[1]
                and 0 <= oz < grid_shape[2]
            ):
                grid[ox, oy, oz] += scaled_occ
                if grid[ox, oy, oz] > log_odds_clamp:
                    grid[ox, oy, oz] = log_odds_clamp
    else:
        ex = int((endpoint[0] - bounds[0, 0]) / resolution)
        ey = int((endpoint[1] - bounds[1, 0]) / resolution)
        ez = int((endpoint[2] - bounds[2, 0]) / resolution)

        if (
            0 <= ex < grid_shape[0]
            and 0 <= ey < grid_shape[1]
            and 0 <= ez < grid_shape[2]
        ):
            grid[ex, ey, ez] += scaled_occ
            if grid[ex, ey, ez] > log_odds_clamp:
                grid[ex, ey, ez] = log_odds_clamp


@njit
def _unproject_and_update_depth_numba(
    grid: np.ndarray,
    cam_origin: np.ndarray,
    rot_matrix: np.ndarray,
    depth_image: np.ndarray,
    segmentation: np.ndarray,
    object_ids: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    bounds: np.ndarray,
    resolution: float,
    free_update: float,
    occ_update: float,
    log_odds_clamp: float,
    stride: int,
    free_space_margin: float,
    object_thickness: float,
    dense_stride: int,
    dense_window: int,
    depth_noise_scale: float,
) -> None:
    """Unproject depth pixels with segmentation-aware adaptive sampling."""
    height, width = depth_image.shape
    n_objects = len(object_ids)

    object_pixels_v = np.zeros(10000, dtype=np.int32)
    object_pixels_u = np.zeros(10000, dtype=np.int32)
    n_object_pixels = 0

    for v in range(0, height, stride):
        for u in range(0, width, stride):
            d = depth_image[v, u]
            if d <= 0 or not np.isfinite(d):
                continue

            seg_id = segmentation[v, u]
            is_object = False
            for i in range(n_objects):
                if seg_id == object_ids[i]:
                    is_object = True
                    if n_object_pixels < 10000:
                        object_pixels_v[n_object_pixels] = v
                        object_pixels_u[n_object_pixels] = u
                        n_object_pixels += 1
                    break

            x_cam = (u - cx) * d / fx
            y_cam = (v - cy) * d / fy
            point_cam = np.array([x_cam, y_cam, d], dtype=np.float32)
            point_world = rot_matrix @ point_cam + cam_origin

            _update_ray_numba(
                grid,
                cam_origin,
                point_world,
                bounds,
                resolution,
                free_update,
                occ_update,
                log_odds_clamp,
                free_space_margin,
                is_object,
                object_thickness,
                depth_noise_scale,
            )

    for idx in range(n_object_pixels):
        cv = object_pixels_v[idx]
        cu = object_pixels_u[idx]
        v_start = max(0, cv - dense_window)
        v_end = min(height, cv + dense_window)
        u_start = max(0, cu - dense_window)
        u_end = min(width, cu + dense_window)

        for v in range(v_start, v_end, dense_stride):
            for u in range(u_start, u_end, dense_stride):
                if (v % stride == 0) and (u % stride == 0):
                    continue

                d = depth_image[v, u]
                if d <= 0 or not np.isfinite(d):
                    continue

                seg_id = segmentation[v, u]
                is_object = False
                for i in range(n_objects):
                    if seg_id == object_ids[i]:
                        is_object = True
                        break

                x_cam = (u - cx) * d / fx
                y_cam = (v - cy) * d / fy
                point_cam = np.array([x_cam, y_cam, d], dtype=np.float32)
                point_world = rot_matrix @ point_cam + cam_origin

                _update_ray_numba(
                    grid,
                    cam_origin,
                    point_world,
                    bounds,
                    resolution,
                    free_update,
                    occ_update,
                    log_odds_clamp,
                    free_space_margin,
                    is_object,
                    object_thickness,
                    depth_noise_scale,
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
        thresholds: BeliefConfig | None = None,
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
        self.thresholds = thresholds or BeliefConfig()

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
        segmentation: np.ndarray,
        object_ids: list[int],
        camera_intrinsics: CameraIntrinsics,
        ego_pose_confidence: float = 1.0,
        stride: int = 10,
        free_space_margin: float = 0.05,
        free_update: float = 0.3,
        occ_update: float = 2.0,
        object_thickness: float = 0.05,
        dense_stride: int = 2,
        dense_window: int = 15,
        depth_noise_scale: float = 0.02,
    ) -> None:
        """Update occupancy grid with segmentation-aware ray-casting.

        Args:
            camera_pose: ((x,y,z), (qx,qy,qz,qw)) in world frame
            depth_image: (H, W) depth values in meters
            segmentation: (H, W) object IDs per pixel
            object_ids: List of object IDs to track
            camera_intrinsics: Camera parameters
            ego_pose_confidence: Weight for updates
            stride: Base pixel sampling stride
            free_space_margin: Buffer zone before depth endpoint
            free_update: Log-odds decrement for free space (L_FREE)
            occ_update: Log-odds increment for occupied (L_OCC)
            object_thickness: Depth to mark as occupied behind object surface
            dense_stride: Stride for dense sampling near objects
            dense_window: Pixel window for dense sampling around objects
            depth_noise_scale: Scaling per meter of depth (handles noise at distance)
        """
        cam_pos, cam_quat = camera_pose
        cam_origin = np.array(cam_pos, dtype=np.float32)

        rot_matrix = p.getMatrixFromQuaternion(cam_quat)
        rot_matrix = np.array(rot_matrix, dtype=np.float32).reshape((3, 3))

        scaled_free = free_update * ego_pose_confidence
        scaled_occ = occ_update * ego_pose_confidence

        _unproject_and_update_depth_numba(
            self.grid,
            cam_origin,
            rot_matrix,
            depth_image,
            segmentation.astype(np.int32),
            np.array(object_ids, dtype=np.int32),
            float(camera_intrinsics.fx),
            float(camera_intrinsics.fy),
            float(camera_intrinsics.cx),
            float(camera_intrinsics.cy),
            self.bounds,
            self.resolution,
            scaled_free,
            scaled_occ,
            self.log_odds_clamp,
            stride,
            free_space_margin,
            object_thickness,
            dense_stride,
            dense_window,
            depth_noise_scale,
        )

    def get_occupancy_probabilities(self) -> np.ndarray:
        """Convert log-odds to probabilities."""
        return 1.0 / (1.0 + np.exp(-self.grid))

    def get_free_voxels(self) -> list[tuple[float, float, float]]:
        """Return xyz coordinates of voxels with occupancy below free
        threshold."""
        probs = self.get_occupancy_probabilities()
        free_mask = probs < self.thresholds.free_max

        free_coords = []
        for ix, iy, iz in zip(*np.where(free_mask)):
            xyz = self._voxel_to_xyz((ix, iy, iz))
            free_coords.append(xyz)

        return free_coords

    def get_occupied_voxels(
        self, occupied_min: float | None = None
    ) -> list[tuple[float, float, float]]:
        """Return xyz coordinates of voxels with occupancy above occupied
        threshold."""
        if occupied_min is None:
            occupied_min = self.thresholds.occupied_min
        probs = self.get_occupancy_probabilities()
        occupied_mask = probs > occupied_min

        coords = []
        for ix, iy, iz in zip(*np.where(occupied_mask)):
            xyz = self._voxel_to_xyz((int(ix), int(iy), int(iz)))
            coords.append(xyz)

        return coords

    def is_occupied(self, xyz: tuple[float, float, float]) -> bool:
        """Check if position is occupied (prob above occupied threshold)."""
        voxel = self._xyz_to_voxel(xyz)
        if not self._is_valid_voxel(voxel):
            return False

        prob = 1.0 / (1.0 + np.exp(-self.grid[voxel]))
        return bool(prob > self.thresholds.occupied_min)

    def is_free(self, xyz: tuple[float, float, float]) -> bool:
        """Check if position is free (prob below free threshold)."""
        voxel = self._xyz_to_voxel(xyz)
        if not self._is_valid_voxel(voxel):
            return False

        prob = 1.0 / (1.0 + np.exp(-self.grid[voxel]))
        return bool(prob < self.thresholds.free_max)

    def get_unobserved_voxels(
        self,
        free_max: float | None = None,
        occupied_min: float | None = None,
    ) -> list[tuple[float, float, float]]:
        """Return xyz of voxels with occupancy in the uncertain range."""
        if free_max is None:
            free_max = self.thresholds.free_max
        if occupied_min is None:
            occupied_min = self.thresholds.occupied_min
        probs = self.get_occupancy_probabilities()
        unobserved_mask = (probs >= free_max) & (probs <= occupied_min)

        coords = []
        for ix, iy, iz in zip(*np.where(unobserved_mask)):
            xyz = self._voxel_to_xyz((int(ix), int(iy), int(iz)))
            coords.append(xyz)

        return coords

    def visualize(
        self,
        physics_client_id: int,
        free_max: float | None = None,
        occupied_min: float | None = None,
        color: tuple[float, float, float] = (0.3, 0.5, 1.0),
        point_size: float = 3.0,
        max_voxels: int = 2000,
        z_range: tuple[float, float] | None = (0.0, 0.06),
    ) -> list[int]:
        """Draw unobserved voxels as points (batched for speed)."""
        unobserved = self.get_unobserved_voxels(
            free_max=free_max, occupied_min=occupied_min
        )

        if z_range is not None:
            unobserved = [
                (x, y, z) for x, y, z in unobserved if z_range[0] <= z <= z_range[1]
            ]

        if len(unobserved) == 0:
            return []

        if len(unobserved) > max_voxels:
            indices = np.random.choice(len(unobserved), max_voxels, replace=False)
            unobserved = [unobserved[i] for i in indices]

        debug_id = p.addUserDebugPoints(
            unobserved,
            [color] * len(unobserved),
            pointSize=point_size,
            physicsClientId=physics_client_id,
        )
        return [debug_id]

    def visualize_occupied(
        self,
        physics_client_id: int,
        occupied_min: float | None = None,
        color: tuple[float, float, float] = (1.0, 0.2, 0.2),
        point_size: float = 5.0,
        max_voxels: int = 1000,
        z_range: tuple[float, float] | None = (0.0, 0.1),
    ) -> list[int]:
        """Draw occupied voxels as points (red by default, batched for
        speed)."""
        occupied = self.get_occupied_voxels(occupied_min=occupied_min)

        if z_range is not None:
            occupied = [
                (x, y, z) for x, y, z in occupied if z_range[0] <= z <= z_range[1]
            ]

        if len(occupied) == 0:
            return []

        if len(occupied) > max_voxels:
            indices = np.random.choice(len(occupied), max_voxels, replace=False)
            occupied = [occupied[i] for i in indices]

        debug_id = p.addUserDebugPoints(
            occupied,
            [color] * len(occupied),
            pointSize=point_size,
            physicsClientId=physics_client_id,
        )
        return [debug_id]

    @staticmethod
    def clear_visualization(debug_ids: list[int], physics_client_id: int) -> None:
        """Remove previously drawn debug items."""
        for debug_id in debug_ids:
            p.removeUserDebugItem(debug_id, physicsClientId=physics_client_id)
