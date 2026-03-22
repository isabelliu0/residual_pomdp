"""TabletopViewOcclusionEnv: pick task where target may be occluded from camera."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from tomsgeoms2d.structs import Circle

from residual_controllers.envs.tabletop_base import TabletopBaseEnv


@dataclass
class TabletopViewOcclusionScene:
    """Scene data for TabletopViewOcclusionEnv."""

    robot: FingeredSingleArmPyBulletRobot
    table_id: int
    object_ids: list[int]
    object_colors: list[tuple[float, float, float, float]]
    target_idx: int
    target_area_id: int
    physics_client_id: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


@dataclass(frozen=True)
class TabletopViewOcclusionEnvState:
    """State for env/sim synchronization."""

    robot_joints: tuple[float, ...]
    object_poses: tuple[Pose, ...]
    held_object_idx: int | None
    grasp_transform: Pose | None
    target_area_pose: Pose | None = None


class TabletopViewOcclusionEnv(TabletopBaseEnv):
    """Tabletop pick task where the target block may be occluded from the wrist
    camera by other objects placed in front of it."""

    def __init__(
        self,
        gui: bool = False,
        num_objects: int = 5,
        occlusion_prob: float = 0.7,
        camera_width: int = 640,
        camera_height: int = 480,
        render_mode: str | None = None,
    ) -> None:
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        self._scene: TabletopViewOcclusionScene | None = None
        self._object_ids: list[int] = []
        self._object_colors: list[tuple[float, float, float, float]] = []
        self._target_area_id: int = -1
        self._object_labels: list[str] = []
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        super().__init__(
            gui=gui,
            camera_width=camera_width,
            camera_height=camera_height,
            render_mode=render_mode,
        )

    def _setup_scene(self) -> None:
        _distractor_colors = [
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0, 1.0),
            (0.5, 0.0, 1.0, 1.0),
        ]
        _target_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)

        target_block = create_pybullet_block(
            color=_target_color,
            half_extents=(0.025, 0.025, 0.025),
            physics_client_id=self.physics_client_id,
            mass=0.1,
            friction=0.9,
        )
        set_pose(
            target_block,
            Pose(position=(-10.0, 0.0, 0.0)),
            self.physics_client_id,
        )
        self._object_ids.append(target_block)
        self._object_colors.append(_target_color)

        for i in range(self.num_objects - 1):
            color = _distractor_colors[i % len(_distractor_colors)]
            obj = create_pybullet_block(
                color=color,
                half_extents=(0.025, 0.025, 0.025),
                physics_client_id=self.physics_client_id,
                mass=0.1,
                friction=0.9,
            )
            set_pose(
                obj,
                Pose(position=(-10.0, float(i + 1), 0.0)),
                self.physics_client_id,
            )
            self._object_ids.append(obj)
            self._object_colors.append(color)

        self._target_area_id = create_pybullet_block(
            color=(0.0, 0.8, 0.0, 0.5),
            half_extents=(0.05, 0.05, 0.002),
            physics_client_id=self.physics_client_id,
            mass=0,
            friction=0.0,
        )
        set_pose(
            self._target_area_id,
            Pose(position=(-10.0, float(self.num_objects), 0.0)),
            self.physics_client_id,
        )

        self._object_labels = [chr(65 + i) for i in range(self.num_objects)]
        self._label_to_id = dict(zip(self._object_labels, self._object_ids))
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}

        max_objects = 10
        self.observation_space = gymnasium.spaces.Dict(
            {
                "joint_positions": gymnasium.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(9,), dtype=np.float32
                ),
                "camera_pose": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "object_poses": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_objects, 7), dtype=np.float32
                ),
                "held_object_idx": gymnasium.spaces.Box(
                    low=-1, high=max_objects, shape=(1,), dtype=np.int32
                ),
                "grasp_transform": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "target_area_pose": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    def _reset_scene(self) -> None:
        self._sample_target_pose_polar(
            self._object_ids[0],
            r_min=0.25,
            r_max=0.35,
            z=0.025,
            collision_check_ids=[],
        )
        placed_ids = [self._object_ids[0]]

        for i in range(1, self.num_objects):
            self._sample_free_object_pose(
                self._object_ids[i],
                x_range=(0.3, 0.7),
                y_range=(-0.4, 0.4),
                z=0.025,
                collision_check_ids=placed_ids.copy(),
            )
            placed_ids.append(self._object_ids[i])

        set_pose(
            self._target_area_id,
            Pose(position=(0.65, 0.0, 0.002), orientation=(0, 0, 0, 1)),
            self.physics_client_id,
        )

        self._scene = TabletopViewOcclusionScene(
            robot=self.robot,
            table_id=self._table_id,
            object_ids=self._object_ids,
            object_colors=self._object_colors,
            target_idx=0,
            target_area_id=self._target_area_id,
            physics_client_id=self.physics_client_id,
            label_to_id=self._label_to_id,
            id_to_label=self._id_to_label,
        )

    def _get_label_to_id(self) -> dict[str, int]:
        return self._label_to_id

    def _get_movable_object_ids(self) -> list[int]:
        return self._object_ids

    def _get_excluded_aabbs(self) -> list[tuple[float, float, float, float]]:
        area_aabb = p.getAABB(
            self._target_area_id, physicsClientId=self.physics_client_id
        )
        return [
            (
                float(area_aabb[0][0]),
                float(area_aabb[0][1]),
                float(area_aabb[1][0]),
                float(area_aabb[1][1]),
            )
        ]

    def get_observation(self) -> dict[str, np.ndarray]:
        assert self.robot is not None
        assert self._scene is not None

        joint_positions = self.robot.get_joint_positions()

        camera_link_id = self.robot.end_effector_id
        camera_pose = get_link_pose(
            self.robot.robot_id,
            camera_link_id,
            self.physics_client_id,
        )
        camera_pose_array = np.concatenate(
            [np.array(camera_pose.position), np.array(camera_pose.orientation)]
        )

        object_poses = np.zeros((10, 7), dtype=np.float32)
        for i, obj_id in enumerate(self._scene.object_ids):
            pose = get_pose(obj_id, self.physics_client_id)
            object_poses[i, :3] = pose.position
            object_poses[i, 3:] = pose.orientation

        held_idx = -1
        if self._held_object_id is not None:
            held_idx = self._scene.object_ids.index(self._held_object_id)

        grasp_tf = np.zeros(7, dtype=np.float32)
        if self._grasp_transform is not None:
            grasp_tf[:3] = self._grasp_transform.position
            grasp_tf[3:] = self._grasp_transform.orientation

        area_pose = get_pose(self._scene.target_area_id, self.physics_client_id)
        target_area_pose_arr = np.concatenate(
            [np.array(area_pose.position), np.array(area_pose.orientation)]
        ).astype(np.float32)

        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "camera_pose": camera_pose_array.astype(np.float32),
            "object_poses": object_poses,
            "held_object_idx": np.array([held_idx], dtype=np.int32),
            "grasp_transform": grasp_tf,
            "target_area_pose": target_area_pose_arr,
        }

    def _get_terminated(self) -> bool:
        return self._held_object_id is None and self._is_target_in_area()

    def _get_reward(self) -> float:
        return 1.0 if self._get_terminated() else 0.0

    def _get_info(self) -> dict:
        return {"target_object_label": "A"}

    def get_collision_ids(self) -> set[int]:
        return {self._table_id} | set(self._object_ids)

    @property
    def scene(self) -> TabletopViewOcclusionScene | None:
        return self._scene

    @property
    def object_labels(self) -> list[str]:
        """Get the list of object labels in the scene, ordered by their IDs."""
        return self._object_labels

    def get_state(self) -> TabletopViewOcclusionEnvState:
        """Get the current state of the environment."""
        assert self.robot is not None
        assert self._scene is not None

        robot_joints = tuple(self.robot.get_joint_positions())
        object_poses = tuple(
            get_pose(obj_id, self.physics_client_id)
            for obj_id in self._scene.object_ids
        )
        target_area_pose = get_pose(self._scene.target_area_id, self.physics_client_id)

        held_object_idx = None
        if self._held_object_id is not None:
            held_object_idx = self._scene.object_ids.index(self._held_object_id)

        return TabletopViewOcclusionEnvState(
            robot_joints=robot_joints,
            object_poses=object_poses,
            held_object_idx=held_object_idx,
            grasp_transform=self._grasp_transform,
            target_area_pose=target_area_pose,
        )

    def set_state(self, state: TabletopViewOcclusionEnvState) -> None:
        """Set the environment state to the given state."""
        assert self.robot is not None
        assert self._scene is not None

        joints = list(state.robot_joints)
        if len(joints) >= 9:
            finger_avg = (joints[7] + joints[8]) / 2
            joints[7] = finger_avg
            joints[8] = finger_avg
        self.robot.set_joints(joints)

        for i, pose in enumerate(state.object_poses):
            set_pose(self._scene.object_ids[i], pose, self.physics_client_id)

        if state.target_area_pose is not None:
            set_pose(
                self._scene.target_area_id,
                state.target_area_pose,
                self.physics_client_id,
            )

        if state.held_object_idx is not None:
            self._held_object_id = self._scene.object_ids[state.held_object_idx]
            self._grasp_transform = state.grasp_transform
        else:
            self._held_object_id = None
            self._grasp_transform = None

    def _is_target_in_area(self) -> bool:
        assert self._scene is not None
        target_id = self._scene.object_ids[self._scene.target_idx]
        target_aabb = p.getAABB(target_id, physicsClientId=self.physics_client_id)
        area_aabb = p.getAABB(
            self._scene.target_area_id, physicsClientId=self.physics_client_id
        )
        return (
            target_aabb[0][0] >= area_aabb[0][0]
            and target_aabb[1][0] <= area_aabb[1][0]
            and target_aabb[0][1] >= area_aabb[0][1]
            and target_aabb[1][1] <= area_aabb[1][1]
        )

    def _euler_to_quat(
        self, roll: float, pitch: float, yaw: float
    ) -> tuple[float, float, float, float]:
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _sample_target_pose_polar(
        self,
        object_id: int,
        r_min: float,
        r_max: float,
        z: float,
        collision_check_ids: list[int],
        angle_min: float = -np.pi / 3,
        angle_max: float = np.pi / 3,
        max_attempts: int = 1000,
    ) -> Pose:
        exclusion_zone = Circle(0.0, 0.0, r_min)
        for _ in range(max_attempts):
            if np.random.random() < 0.5:
                theta = np.random.uniform(np.pi / 6, angle_max)
            else:
                theta = np.random.uniform(angle_min, -np.pi / 6)
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2))
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            assert not exclusion_zone.contains_point(x, y)

            quat = self._euler_to_quat(0, 0, np.random.uniform(0, 2 * np.pi))
            pose = Pose(position=(x, y, z), orientation=quat)
            set_pose(object_id, pose, self.physics_client_id)

            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            collision_free = all(
                not check_body_collisions(
                    object_id, cid, self.physics_client_id, distance_threshold=1e-3
                )
                for cid in collision_check_ids
            )
            if collision_free:
                return pose

        raise RuntimeError(
            f"Could not sample free target position after {max_attempts} attempts."
        )

    def _sample_free_object_pose(
        self,
        object_id: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z: float,
        collision_check_ids: list[int],
        max_attempts: int = 1000,
    ) -> Pose:
        for _ in range(max_attempts):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            quat = self._euler_to_quat(0, 0, np.random.uniform(0, 2 * np.pi))
            pose = Pose(position=(x, y, z), orientation=quat)
            set_pose(object_id, pose, self.physics_client_id)

            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            collision_free = all(
                not check_body_collisions(
                    object_id, cid, self.physics_client_id, distance_threshold=1e-3
                )
                for cid in collision_check_ids
            )
            if collision_free:
                return pose

        raise RuntimeError(
            f"Could not sample free object position after {max_attempts} attempts."
        )
