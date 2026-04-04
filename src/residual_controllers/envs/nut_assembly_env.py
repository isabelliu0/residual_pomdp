"""NutAssemblyEnv: robot must pick up a round nut and lower it onto a round peg."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

from residual_controllers.beliefs.structs import BeliefConfig, TabletopState
from residual_controllers.envs.tabletop_base import TabletopBaseEnv

_ASSETS_DIR = Path(__file__).parent / "assets"
_NUT_URDF_PATH = _ASSETS_DIR / "round-nut.urdf"
_TEXTURE_PATH = _ASSETS_DIR / "steel-scratched.png"

_PEG_RADIUS = 0.015  # nut inner radius ~0.0156m -> ~0.6mm clearance each side
_PEG_HEIGHT = 0.1
_NUT_HALF_HEIGHT = 0.005


def _load_nut(physics_client_id: int) -> int:
    nut_id = p.loadURDF(
        str(_NUT_URDF_PATH),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=physics_client_id,
    )
    try:
        tex_id = p.loadTexture(str(_TEXTURE_PATH))
        p.changeVisualShape(
            nut_id, -1, textureUniqueId=tex_id, physicsClientId=physics_client_id
        )
    except Exception:  # pylint: disable=broad-except
        pass
    return nut_id


def _load_peg(physics_client_id: int) -> int:
    pcid = physics_client_id
    col_id = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=_PEG_RADIUS, height=_PEG_HEIGHT, physicsClientId=pcid
    )
    vis_id = p.createVisualShape(
        p.GEOM_CYLINDER, radius=_PEG_RADIUS, length=_PEG_HEIGHT, physicsClientId=pcid
    )
    peg_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.0, 0.0, 0.0],
        physicsClientId=pcid,
    )
    try:
        tex_id = p.loadTexture(str(_TEXTURE_PATH))
        p.changeVisualShape(peg_id, -1, textureUniqueId=tex_id, physicsClientId=pcid)
    except Exception:  # pylint: disable=broad-except
        pass
    return peg_id


@dataclass
class NutAssemblyScene:
    """Scene data for NutAssemblyEnv."""

    robot: FingeredSingleArmPyBulletRobot
    table_id: int
    object_ids: list[int]
    nut_idx: int
    peg_idx: int
    physics_client_id: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


@dataclass(frozen=True)
class NutAssemblyEnvState:
    """State for env/sim synchronization."""

    robot_joints: tuple[float, ...]
    object_poses: tuple[Pose, ...]
    held_object_idx: int | None
    grasp_transform: Pose | None


class NutAssemblyEnv(TabletopBaseEnv):
    """Tabletop task: pick up the round nut and lower it onto the fixed peg."""

    def __init__(
        self,
        gui: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        render_mode: str | None = None,
    ) -> None:
        self._scene: NutAssemblyScene | None = None
        self._nut_id: int = -1
        self._peg_id: int = -1
        self._peg_center: tuple[float, float, float] = (0.5, 0.0, _PEG_HEIGHT / 2)
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
        self._nut_id = _load_nut(self.physics_client_id)
        set_pose(
            self._nut_id,
            Pose(position=(-10.0, 0.0, _NUT_HALF_HEIGHT)),
            self.physics_client_id,
        )

        self._peg_id = _load_peg(self.physics_client_id)
        self._peg_center = (0.5, 0.0, _PEG_HEIGHT / 2)
        set_pose(
            self._peg_id,
            Pose(position=self._peg_center),
            self.physics_client_id,
        )

        self._object_ids = [self._nut_id, self._peg_id]
        self._label_to_id = {"NUT": self._nut_id, "PEG": self._peg_id}
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

    def _sample_nut_pose(self) -> Pose:
        peg_cx, peg_cy = self._peg_center[0], self._peg_center[1]
        r = float(np.random.uniform(0.1, 0.25))
        if np.random.random() < 0.5:
            theta = float(np.random.uniform(np.pi / 3, np.pi / 2))
        else:
            theta = float(np.random.uniform(-np.pi / 2, -np.pi / 3))
        x = peg_cx + r * np.cos(theta)
        y = peg_cy + r * np.sin(theta)
        return Pose(position=(x, y, _NUT_HALF_HEIGHT), orientation=(0, 0, 0, 1))

    def _reset_scene(self) -> None:
        nut_pose = self._sample_nut_pose()
        set_pose(
            self._nut_id,
            nut_pose,
            self.physics_client_id,
        )
        set_pose(
            self._peg_id,
            Pose(position=self._peg_center, orientation=(0, 0, 0, 1)),
            self.physics_client_id,
        )
        self._scene = NutAssemblyScene(
            robot=self.robot,
            table_id=self._table_id,
            object_ids=self._object_ids,
            nut_idx=0,
            peg_idx=1,
            physics_client_id=self.physics_client_id,
            label_to_id=self._label_to_id,
            id_to_label=self._id_to_label,
        )

    def _get_label_to_id(self) -> dict[str, int]:
        return self._label_to_id

    def get_belief_config(self) -> BeliefConfig:
        return BeliefConfig(label_n_fold={"NUT": 36, "PEG": 36})

    def _get_movable_object_ids(self) -> list[int]:
        return [self._nut_id]

    def _get_excluded_aabbs(self) -> list[tuple[float, float, float, float]]:
        return []

    def get_observation(self) -> dict[str, np.ndarray]:
        assert self.robot is not None
        assert self._scene is not None

        joint_positions = self.robot.get_joint_positions()
        camera_link_id = self.robot.end_effector_id
        camera_pose = get_link_pose(
            self.robot.robot_id, camera_link_id, self.physics_client_id
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
        return self._is_assembled()

    def _get_reward(self) -> float:
        return 1.0 if self._get_terminated() else 0.0

    def _get_info(self) -> dict:
        return {"target_object_label": "NUT"}

    def get_collision_ids(self) -> set[int]:
        return {self._table_id, self._nut_id, self._peg_id}

    @property
    def scene(self) -> NutAssemblyScene | None:
        return self._scene

    def get_state(self) -> NutAssemblyEnvState:
        """Get complete state for env/sim synchronization."""
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
        return NutAssemblyEnvState(
            robot_joints=robot_joints,
            object_poses=object_poses,
            held_object_idx=held_object_idx,
            grasp_transform=self._grasp_transform,
        )

    def set_state(self, state: NutAssemblyEnvState) -> None:
        """Set complete state for env/sim synchronization."""
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

    def _update_held_object_pose(self) -> None:
        """Move the held nut to the desired EE-relative pose, then push it out
        of any penetrations with the peg and table so the nut slides along
        surfaces rather than passing through them."""
        assert self._held_object_id is not None
        assert self._grasp_transform is not None

        ee_pose = self.robot.get_end_effector_pose()
        desired_pose = multiply_poses(ee_pose, self._grasp_transform)
        set_pose(self._held_object_id, desired_pose, self.physics_client_id)

        p.performCollisionDetection(physicsClientId=self.physics_client_id)
        correction = np.zeros(3)
        for other_id in (self._peg_id, self._table_id):
            contacts = p.getContactPoints(
                self._held_object_id,
                other_id,
                physicsClientId=self.physics_client_id,
            )
            for contact in contacts or []:
                depth = float(contact[8])  # negative = penetrating
                if depth < 0:
                    # contact[7] is the contact normal on bodyB pointing toward bodyA
                    normal = np.array(contact[7], dtype=float)
                    correction += -depth * normal

        if np.any(correction != 0):
            resolved_pos = np.array(desired_pose.position) + correction
            resolved_pose = Pose(
                position=tuple(resolved_pos), orientation=desired_pose.orientation
            )
            set_pose(self._held_object_id, resolved_pose, self.physics_client_id)
            # Keep grasp transform consistent with the displaced nut position
            self._grasp_transform = multiply_poses(ee_pose.invert(), resolved_pose)

    def _update_belief_from_contact(self) -> None:
        """Collapse belief particles when nut is fitted onto the peg while
        held.

        When the nut hole aligns with the peg shaft (xy within peg
        clearance and nut z at or below peg top), the contact geometry
        gives near-certain localization of both objects.
        """
        if self.belief is None or self._held_object_id != self._nut_id:
            return
        nut_pos = np.array(get_pose(self._nut_id, self.physics_client_id).position)
        peg_aabb = p.getAABB(self._peg_id, physicsClientId=self.physics_client_id)
        peg_cx = (float(peg_aabb[0][0]) + float(peg_aabb[1][0])) / 2
        peg_cy = (float(peg_aabb[0][1]) + float(peg_aabb[1][1])) / 2
        peg_top_z = float(peg_aabb[1][2])
        xy_dist = float(np.linalg.norm(nut_pos[:2] - np.array([peg_cx, peg_cy])))
        if xy_dist > 0.005 or nut_pos[2] > peg_top_z + 0.002:
            return
        nut_gt = get_pose(self._nut_id, self.physics_client_id)
        peg_gt = get_pose(self._peg_id, self.physics_client_id)
        nut_pose_gt: tuple = nut_gt.position + nut_gt.orientation
        peg_pose_gt: tuple = peg_gt.position + peg_gt.orientation
        new_particles = []
        for particle in self.belief.particles:
            new_poses = dict(particle.object_poses)
            new_poses["NUT"] = nut_pose_gt
            new_poses["PEG"] = peg_pose_gt
            new_particles.append(
                TabletopState(
                    joint_positions=particle.joint_positions,
                    gripper_open=particle.gripper_open,
                    object_poses=new_poses,
                )
            )
        self.belief.particles = new_particles
        n = len(new_particles)
        self.belief.weights = np.ones(n) / n

    def _is_assembled(self) -> bool:
        """Nut is not held and rests on the table centered around the peg."""
        assert self._scene is not None
        if self._held_object_id == self._nut_id:
            return False
        nut_pos = np.array(get_pose(self._nut_id, self.physics_client_id).position)
        peg_aabb = p.getAABB(self._peg_id, physicsClientId=self.physics_client_id)
        peg_cx = (float(peg_aabb[0][0]) + float(peg_aabb[1][0])) / 2
        peg_cy = (float(peg_aabb[0][1]) + float(peg_aabb[1][1])) / 2
        xy_dist = float(np.linalg.norm(nut_pos[:2] - np.array([peg_cx, peg_cy])))
        return xy_dist < 0.03 and abs(nut_pos[2] - _NUT_HALF_HEIGHT) < 0.015
