"""TabletopObjectOcclusionEnv: milk carton hidden behind cereal box; robot
must pour milk into a cup."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from tomsgeoms2d.structs import Circle

from residual_controllers.envs.objaverse_utils import (
    get_resting_z,
    load_objaverse_object,
    setup_ssl_context,
)
from residual_controllers.envs.tabletop_base import TabletopBaseEnv

_CEREAL_BOX_UID = "18501d8c14144a9492b50cb4f99fb6eb"
_MILK_CARTON_UID = "caaee8a82cfd4579bf8ea94fa7be6b54"
_CUP_UID = "89938b8ecedf4ab89d78fd9f4b40b2a4"

_CEREAL_BOX_SCALE = 1.2
_MILK_CARTON_SCALE = 0.06
_CUP_SCALE = 0.06


@dataclass
class TabletopObjectOcclusionScene:
    """Scene data for TabletopObjectOcclusionEnv."""

    robot: FingeredSingleArmPyBulletRobot
    table_id: int
    object_ids: list[int]
    cereal_box_idx: int
    milk_carton_idx: int
    cup_idx: int
    physics_client_id: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


@dataclass(frozen=True)
class TabletopObjectOcclusionEnvState:
    """State for env/sim synchronization."""

    robot_joints: tuple[float, ...]
    object_poses: tuple[Pose, ...]
    held_object_idx: int | None
    grasp_transform: Pose | None


class TabletopObjectOcclusionEnv(TabletopBaseEnv):
    """Tabletop task where the milk carton is fully occluded from the wrist
    camera by a large cereal box, and the robot must pour milk into a cup."""

    def __init__(
        self,
        gui: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        render_mode: str | None = None,
    ) -> None:
        self._scene: TabletopObjectOcclusionScene | None = None
        self._cereal_box_id: int = -1
        self._milk_carton_id: int = -1
        self._cup_id: int = -1
        self._cereal_box_z: float = 0.0
        self._milk_carton_z_tipped: float = 0.0
        self._cup_z: float = 0.0
        self._object_ids: list[int] = []
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        super().__init__(
            gui=gui,
            camera_width=camera_width,
            camera_height=camera_height,
            render_mode=render_mode,
        )

    def _setup_scene(self) -> None:
        setup_ssl_context()

        self._cereal_box_id = load_objaverse_object(
            uid=_CEREAL_BOX_UID,
            scale=_CEREAL_BOX_SCALE,
            mass=0.3,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self._cereal_box_id,
            Pose(position=(-10.0, 0.0, 0.0)),
            self.physics_client_id,
        )
        self._cereal_box_z = get_resting_z(
            self._cereal_box_id, 0.0, self.physics_client_id
        )

        self._milk_carton_id = load_objaverse_object(
            uid=_MILK_CARTON_UID,
            scale=_MILK_CARTON_SCALE,
            mass=0.5,
            physics_client_id=self.physics_client_id,
        )
        _milk_tipped_quat = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
        set_pose(
            self._milk_carton_id,
            Pose(position=(-10.0, 1.0, 0.0), orientation=_milk_tipped_quat),
            self.physics_client_id,
        )
        self._milk_carton_z_tipped = get_resting_z(
            self._milk_carton_id, 0.0, self.physics_client_id
        )

        self._cup_id = load_objaverse_object(
            uid=_CUP_UID,
            scale=_CUP_SCALE,
            mass=0.1,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self._cup_id,
            Pose(position=(-10.0, 2.0, 0.0)),
            self.physics_client_id,
        )
        self._cup_z = get_resting_z(self._cup_id, 0.0, self.physics_client_id)

        self._object_ids = [
            self._cereal_box_id,
            self._milk_carton_id,
            self._cup_id,
        ]
        self._label_to_id = {
            "CEREAL": self._cereal_box_id,
            "MILK": self._milk_carton_id,
            "CUP": self._cup_id,
        }
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}

        self.observation_space = gymnasium.spaces.Dict(
            {
                "joint_positions": gymnasium.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(9,), dtype=np.float32
                ),
                "camera_pose": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "object_poses": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10, 7), dtype=np.float32
                ),
                "held_object_idx": gymnasium.spaces.Box(
                    low=-1, high=10, shape=(1,), dtype=np.int32
                ),
                "grasp_transform": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    def _reset_scene(self) -> None:
        set_pose(
            self._cereal_box_id,
            Pose(position=(0.40, 0.0, self._cereal_box_z), orientation=(0, 0, 0, 1)),
            self.physics_client_id,
        )
        set_pose(
            self._milk_carton_id,
            Pose(
                position=(0.55, 0.0, self._milk_carton_z_tipped),
                orientation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            ),
            self.physics_client_id,
        )
        milk_center = Pose(position=(0.55, 0.0, self._cup_z))
        self._sample_target_pose_polar(
            center=milk_center,
            object_id=self._cup_id,
            r=0.25,
            z=self._cup_z,
            collision_check_ids=[self._cereal_box_id, self._milk_carton_id],
        )

        self._scene = TabletopObjectOcclusionScene(
            robot=self.robot,
            table_id=self._table_id,
            object_ids=self._object_ids,
            cereal_box_idx=0,
            milk_carton_idx=1,
            cup_idx=2,
            physics_client_id=self.physics_client_id,
            label_to_id=self._label_to_id,
            id_to_label=self._id_to_label,
        )

    def _get_label_to_id(self) -> dict[str, int]:
        return self._label_to_id

    def _get_movable_object_ids(self) -> list[int]:
        return self._object_ids

    def _get_excluded_aabbs(self) -> list[tuple[float, float, float, float]]:
        return []

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

        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "camera_pose": camera_pose_array.astype(np.float32),
            "object_poses": object_poses,
            "held_object_idx": np.array([held_idx], dtype=np.int32),
            "grasp_transform": grasp_tf,
        }

    def _get_terminated(self) -> bool:
        return self._is_pouring()

    def _get_reward(self) -> float:
        return 1.0 if self._get_terminated() else 0.0

    def _get_info(self) -> dict:
        return {"target_object_label": "MILK"}

    def _get_grasp_aabb_threshold(self, obj_id: int) -> float:
        if obj_id == self._milk_carton_id:
            return 0.02
        return 1e-4

    def get_collision_ids(self) -> set[int]:
        return {
            self._table_id,
            self._cereal_box_id,
            self._milk_carton_id,
            self._cup_id,
        }

    @property
    def scene(self) -> TabletopObjectOcclusionScene | None:
        return self._scene

    def get_state(self) -> TabletopObjectOcclusionEnvState:
        """Get current env state."""
        assert self.robot is not None
        assert self._scene is not None

        robot_joints = tuple(self.robot.get_joint_positions())
        object_poses = tuple(
            get_pose(obj_id, self.physics_client_id)
            for obj_id in self._scene.object_ids
        )

        held_object_idx = None
        if self._held_object_id is not None:
            held_object_idx = self._scene.object_ids.index(self._held_object_id)

        return TabletopObjectOcclusionEnvState(
            robot_joints=robot_joints,
            object_poses=object_poses,
            held_object_idx=held_object_idx,
            grasp_transform=self._grasp_transform,
        )

    def set_state(self, state: TabletopObjectOcclusionEnvState) -> None:
        """Set env state to given state."""
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

        if state.held_object_idx is not None:
            self._held_object_id = self._scene.object_ids[state.held_object_idx]
            self._grasp_transform = state.grasp_transform
        else:
            self._held_object_id = None
            self._grasp_transform = None

    def _is_pouring(self) -> bool:
        assert self._scene is not None
        milk_id = self._scene.object_ids[self._scene.milk_carton_idx]
        cup_id = self._scene.object_ids[self._scene.cup_idx]
        if self._held_object_id != milk_id:
            return False
        milk_pose = get_pose(milk_id, self.physics_client_id)
        cup_pose = get_pose(cup_id, self.physics_client_id)
        cup_aabb = p.getAABB(cup_id, physicsClientId=self.physics_client_id)
        cup_top_z = float(cup_aabb[1][2])
        milk_aabb = p.getAABB(milk_id, physicsClientId=self.physics_client_id)
        milk_half_y = float(milk_aabb[1][1] - milk_aabb[0][1])
        milk_pos = np.array(milk_pose.position)
        cup_pos = np.array(cup_pose.position)
        horizontal_dist = float(np.linalg.norm(milk_pos[:2] - cup_pos[:2])) / 2
        ee_pose = self.robot.get_end_effector_pose()
        ee_z_world = p.rotateVector(ee_pose.orientation, [0, 0, 1])
        is_tilted = abs(float(ee_z_world[2])) < 0.9
        return (
            horizontal_dist < milk_half_y + 0.05
            and milk_pos[2] > cup_top_z
            and is_tilted
        )

    def _sample_target_pose_polar(
        self,
        center: Pose,
        object_id: int,
        r: float,
        z: float,
        collision_check_ids: list[int],
        angle_min: float = -np.pi / 3,
        angle_max: float = np.pi / 3,
        max_attempts: int = 1000,
    ) -> Pose:
        exclusion_zone = Circle(center.position[0], center.position[1], r - 0.1)
        for _ in range(max_attempts):
            if np.random.random() < 0.5:
                theta = np.random.uniform(angle_max, np.pi / 2)
            else:
                theta = np.random.uniform(-np.pi / 2, angle_min)
            x = center.position[0] + r * np.cos(theta)
            y = center.position[1] + r * np.sin(theta)
            assert not exclusion_zone.contains_point(x, y)

            pose = Pose(position=(x, y, z))
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
            f"Could not sample free cup position after {max_attempts} attempts."
        )
