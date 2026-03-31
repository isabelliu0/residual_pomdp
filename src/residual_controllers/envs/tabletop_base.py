"""Base PyBullet environment for tabletop manipulation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pybullet as p
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection, visualize_pose
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from residual_controllers.beliefs import (
    CameraIntrinsics,
    create_initial_belief,
    predict_belief,
    update_belief,
)
from residual_controllers.beliefs.structs import Belief


@dataclass
class CameraImage:
    """Camera image data."""

    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


class TabletopBaseEnv(gym.Env, ABC):
    """Abstract base for tabletop PyBullet environments."""

    metadata = {"render_modes": ["rgb_array", "rgb_array_external"], "render_fps": 30}

    home_joint_positions = [
        -0.0806406098426434,
        -1.6722951504174777,
        0.07069076842695393,
        -2.7449419709102822,
        0.08184716251979611,
        1.7516337599063168,
        0.7849295270972781,
    ]

    def __init__(
        self,
        gui: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.gui = gui
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.render_mode = render_mode

        if gui:
            self.physics_client_id = create_gui_connection()
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=[0.5, 0.0, 0.15],
                physicsClientId=self.physics_client_id,
            )
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)

        self.wrist_camera_offset = (0.03649966282414946, 0.0, 0.0574)
        self.wrist_camera_quat = (0.0, 0.0, -0.00512551, 0.99996205)

        self.camera_intrinsics = CameraIntrinsics(
            fx=525.0,
            fy=525.0,
            cx=camera_width / 2.0,
            cy=camera_height / 2.0,
            width=camera_width,
            height=camera_height,
        )

        self.robot: FingeredSingleArmPyBulletRobot = cast(
            FingeredSingleArmPyBulletRobot,
            create_pybullet_robot("panda", self.physics_client_id),
        )
        gripper_pos = self.robot.get_joint_positions()[7:]
        self.robot.set_joints(self.home_joint_positions + list(gripper_pos))

        self._table_id = self._create_table()

        self.belief: Belief | None = None
        self._held_object_id: int | None = None
        self._grasp_transform: Pose | None = None
        self._camera_frame_ids: set[int] = set()

        self.action_space = gym.spaces.Box(
            low=np.array([-0.1] * 7 + [-1.0], dtype=np.float32),
            high=np.array([0.1] * 7 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._setup_scene()

    def _create_table(self) -> int:
        table = create_pybullet_block(
            color=(0.6, 0.6, 0.6, 1.0),
            half_extents=(0.75, 0.6, 0.015),
            physics_client_id=self.physics_client_id,
            mass=0,
            friction=0.8,
        )
        set_pose(
            table,
            Pose(position=(0.5, 0.0, -0.015), orientation=(0, 0, 0, 1)),
            self.physics_client_id,
        )
        return table

    @property
    @abstractmethod
    def scene(self) -> Any | None:
        """Current scene data (env-specific dataclass)."""

    @abstractmethod
    def _setup_scene(self) -> None:
        """Create scene-specific PyBullet objects."""

    @abstractmethod
    def _reset_scene(self) -> None:
        """Randomize/place scene objects for a new episode."""

    @abstractmethod
    def _get_label_to_id(self) -> dict[str, int]:
        """Return mapping from object label to PyBullet body ID."""

    @abstractmethod
    def _get_movable_object_ids(self) -> list[int]:
        """Return IDs of all graspable objects."""

    @abstractmethod
    def _get_excluded_aabbs(self) -> list[tuple[float, float, float, float]]:
        """Return 2D AABBs (xmin, ymin, xmax, ymax) excluded from belief
        update."""

    @abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]:
        """Return current observation dict."""

    @abstractmethod
    def _get_terminated(self) -> bool:
        """Return True if the episode is done."""

    @abstractmethod
    def _get_reward(self) -> float:
        """Return current step reward."""

    @abstractmethod
    def get_collision_ids(self) -> set[int]:
        """Return PyBullet IDs for collision checking during motion
        planning."""

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        gripper_pos = self.robot.get_joint_positions()[7:]
        self.robot.set_joints(self.home_joint_positions + list(gripper_pos))

        self._held_object_id = None
        self._grasp_transform = None

        self._reset_scene()

        obs = self.get_observation()
        info = self._get_info()

        camera_image = self.get_camera_image()
        self.belief = create_initial_belief(self, camera_image, num_particles=100)

        if self.gui:
            self._update_camera_visualization()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        joint_delta = action[:7]
        gripper_action = action[7] if len(action) > 7 else 0.0

        if gripper_action > 0.5 and self._held_object_id is not None:
            self._held_object_id = None
            self._grasp_transform = None

        if gripper_action < -0.5 and self._held_object_id is None:
            self._try_grasp()

        current_joints = self.robot.get_joint_positions()
        new_joints = np.array(current_joints[:7]) + joint_delta
        new_joints = np.clip(
            new_joints,
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        )
        self.robot.set_joints(list(new_joints) + list(current_joints[7:]))

        if self._held_object_id is not None and self._grasp_transform is not None:
            self._update_held_object_pose()

        p.stepSimulation(physicsClientId=self.physics_client_id)

        joints = list(self.robot.get_joint_positions())
        avg = (joints[7] + joints[8]) / 2
        joints[7] = avg
        joints[8] = avg
        self.robot.set_joints(joints)

        if self.gui:
            self._update_camera_visualization()

        if self.belief is not None:
            label_to_id = self._get_label_to_id()
            id_to_label = {v: k for k, v in label_to_id.items()}
            held_label = (
                id_to_label.get(self._held_object_id)
                if self._held_object_id is not None
                else None
            )
            self.belief.held_object_label = held_label

            self.belief = predict_belief(
                self.belief,
                joint_delta,
                np.array(self.robot.joint_lower_limits[:7]),
                np.array(self.robot.joint_upper_limits[:7]),
                noise_std=0.01,
            )

            camera_image = self.get_camera_image()
            camera_pose = self.get_camera_pose_se3()
            self.belief = update_belief(
                self.belief,
                camera_image,
                camera_pose,
                self.camera_intrinsics,
                label_to_id,
                self.physics_client_id,
                table_id=self._table_id,
                excluded_aabbs=self._get_excluded_aabbs(),
            )

        obs = self.get_observation()
        terminated = self._get_terminated()
        reward = self._get_reward()
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    @property
    def camera_to_hand_transform(self) -> Pose:
        """Get the fixed transform from the robot's end effector to the wrist
        camera."""
        return Pose(
            position=self.wrist_camera_offset,
            orientation=self.wrist_camera_quat,
        )

    @property
    def camera_to_ee_transform(self) -> Pose:
        """Get the fixed transform from the robot's end effector to the wrist
        camera."""
        return self.camera_to_hand_transform

    def _get_hand_pose(self) -> Pose:
        """Get the current pose of the robot's hand (end effector)."""
        hand_link_id = self.robot.link_from_name("panda_hand")
        return get_link_pose(self.robot.robot_id, hand_link_id, self.physics_client_id)

    def get_camera_pose_se3(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Get the current camera pose as (position, orientation)."""
        hand_pose = self._get_hand_pose()
        camera_pose = multiply_poses(hand_pose, self.camera_to_hand_transform)
        return (camera_pose.position, camera_pose.orientation)

    def get_camera_image(self) -> CameraImage:
        """Capture an image from the wrist camera."""
        hand_pose = self._get_hand_pose()
        camera_pose = multiply_poses(hand_pose, self.camera_to_hand_transform)

        far = 5.0
        rot_matrix = np.array(
            p.getMatrixFromQuaternion(camera_pose.orientation)
        ).reshape((3, 3))
        target_point = np.array(camera_pose.position) + rot_matrix @ np.array(
            [0, 0, far]
        )
        camera_up = tuple(-rot_matrix[:, 1])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pose.position,
            cameraTargetPosition=tuple(target_point),
            cameraUpVector=camera_up,
            physicsClientId=self.physics_client_id,
        )
        fov_y_deg = np.degrees(
            2 * np.arctan(self.camera_height / (2 * self.camera_intrinsics.fy))
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov_y_deg,
            aspect=float(self.camera_width / self.camera_height),
            nearVal=0.02,
            farVal=far,
            physicsClientId=self.physics_client_id,
        )

        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client_id,
        )

        rgb_array = np.array(rgb).reshape((self.camera_height, self.camera_width, 4))[
            :, :, :3
        ]
        depth_array = np.array(depth).reshape((self.camera_height, self.camera_width))
        seg_array = np.array(seg).reshape((self.camera_height, self.camera_width))

        near = 0.02
        depth_array = far * near / (far - (far - near) * depth_array)

        return CameraImage(
            rgb=rgb_array.astype(np.uint8),
            depth=depth_array,
            segmentation=seg_array,
        )

    def _get_grasp_aabb_threshold(self, _obj_id: int) -> float:
        """Squared distance threshold for AABB-based grasp detection."""
        return 1e-4

    def _try_grasp(self) -> None:
        ee_pose = self.robot.get_end_effector_pose()
        ee_pos = np.array(ee_pose.position)
        for obj_id in self._get_movable_object_ids():
            aabb_min, aabb_max = p.getAABB(
                obj_id, physicsClientId=self.physics_client_id
            )
            nearest = np.clip(ee_pos, aabb_min, aabb_max)
            dist_sq = float(np.sum((ee_pos - nearest) ** 2))
            if dist_sq < self._get_grasp_aabb_threshold(obj_id):
                obj_pose = get_pose(obj_id, self.physics_client_id)
                self._held_object_id = obj_id
                self._grasp_transform = multiply_poses(ee_pose.invert(), obj_pose)
                break

    def _update_held_object_pose(self) -> None:
        """Update the pose of the held object to follow the robot's end
        effector."""
        assert self._held_object_id is not None
        assert self._grasp_transform is not None
        ee_pose = self.robot.get_end_effector_pose()
        set_pose(
            self._held_object_id,
            multiply_poses(ee_pose, self._grasp_transform),
            self.physics_client_id,
        )

    def _update_camera_visualization(self) -> None:
        """Update the wrist camera visualization in the GUI."""
        if not self.gui:
            return
        for frame_id in self._camera_frame_ids:
            p.removeUserDebugItem(frame_id, physicsClientId=self.physics_client_id)
        hand_pose = self._get_hand_pose()
        camera_pose = multiply_poses(hand_pose, self.camera_to_hand_transform)
        self._camera_frame_ids = visualize_pose(
            camera_pose,
            self.physics_client_id,
            axis_length=0.1,
        )

    @property
    def held_object_id(self) -> int | None:
        """Get the PyBullet ID of the currently held object."""
        return self._held_object_id

    @held_object_id.setter
    def held_object_id(self, value: int | None) -> None:
        self._held_object_id = value

    @property
    def grasp_transform(self) -> Pose | None:
        """Get the fixed transform from the robot's end effector to the held
        object when grasping."""
        return self._grasp_transform

    @grasp_transform.setter
    def grasp_transform(self, value: Pose | None) -> None:
        self._grasp_transform = value

    def render(self):
        if self.render_mode == "rgb_array":
            return self.get_camera_image().rgb
        if self.render_mode == "rgb_array_external":
            return capture_image(
                self.physics_client_id,
                camera_distance=0.9,
                camera_yaw=90,
                camera_pitch=-25,
                camera_target=(0.5, 0.0, 0.15),
                image_width=640,
                image_height=480,
            )
        return None

    def close(self) -> None:
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
            self.physics_client_id = -1
