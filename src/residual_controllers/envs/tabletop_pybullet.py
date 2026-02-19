"""PyBullet environment for tabletop manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import pybullet as p
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection, visualize_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.link import get_link_pose
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from residual_controllers.beliefs import (
    Belief,
    CameraIntrinsics,
    create_initial_belief,
    predict_belief,
    update_belief,
)


@dataclass
class CameraImage:
    """Camera image data."""

    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


@dataclass
class TabletopScene:
    """Tabletop manipulation scene data."""

    robot: FingeredSingleArmPyBulletRobot
    table_id: int
    object_ids: list[int]
    object_colors: list[tuple[float, float, float, float]]
    target_idx: int
    physics_client_id: int


@dataclass(frozen=True)
class TabletopEnvState:
    """State for env/sim synchronization."""

    robot_joints: tuple[float, ...]
    object_poses: tuple[Pose, ...]
    held_object_idx: int | None
    grasp_transform: Pose | None


class TabletopPickEnv(gym.Env):
    """Tabletop manipulation: robot must pick a target object from a cluttered table."""

    metadata = {"render_modes": ["rgb_array", "rgb_array_external"], "render_fps": 30}

    # Home joint positions from TAMPURA - better wrist camera view for table observation
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
        num_objects: int = 5,
        occlusion_prob: float = 0.7,
        camera_width: int = 640,
        camera_height: int = 480,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.gui = gui
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.render_mode = render_mode

        if gui:
            self.physics_client_id = create_gui_connection()
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.5, 0.0, 0.0],
                physicsClientId=self.physics_client_id,
            )
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)

        self.robot: FingeredSingleArmPyBulletRobot | None = None
        self.scene: TabletopScene | None = None
        self.belief: Belief | None = None
        self._camera_frame_ids: set[int] = set()

        self._held_object_id: int | None = None
        self._grasp_transform: Pose | None = None

        self.wrist_camera_offset = (0.03649966282414946, 0.0, 0.0574)
        self.wrist_camera_quat = (0.0, 0.0, -0.00512551, 0.99996205)
        fx, fy = 525.0, 525.0

        self.camera_intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=camera_width / 2.0,
            cy=camera_height / 2.0,
            width=camera_width,
            height=camera_height,
        )

        max_objects = 10
        self.observation_space = gym.spaces.Dict(
            {
                "joint_positions": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(9,), dtype=np.float32
                ),
                "camera_pose": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "object_poses": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(max_objects, 7), dtype=np.float32
                ),
                "held_object_idx": gym.spaces.Box(
                    low=-1, high=max_objects, shape=(1,), dtype=np.int32
                ),
                "grasp_transform": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-0.1] * 7 + [-1.0], dtype=np.float32),
            high=np.array([0.1] * 7 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )

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

    def _spawn_objects_with_occlusion(self) -> tuple[list[int], list[tuple], int]:
        object_ids: list[int] = []
        colors = []

        target_idx = np.random.randint(0, self.num_objects)
        target_pos: tuple[float, float, float] | None = None

        for i in range(self.num_objects):
            is_target = i == target_idx
            color = (1.0, 0.0, 0.0, 1.0) if is_target else self._get_random_color()

            obj = create_pybullet_block(
                color=color,
                half_extents=(0.025, 0.025, 0.025),
                physics_client_id=self.physics_client_id,
                mass=0.1,
                friction=0.9,
            )

            x_range: tuple[float, float] = (0.3, 0.7)
            y_range: tuple[float, float] = (-0.3, 0.3)

            if is_target:
                if np.random.random() < self.occlusion_prob:
                    y_range = (-0.3, -0.1)
            elif target_pos is not None and np.random.random() < self.occlusion_prob:
                tx, ty = (
                    target_pos[0],  # pylint: disable=unsubscriptable-object
                    target_pos[1],  # pylint: disable=unsubscriptable-object
                )
                x_range = (max(0.3, tx - 0.1), min(0.7, tx + 0.1))
                y_range = (max(-0.3, ty + 0.03), min(0.3, ty + 0.2))

            collision_check_ids = object_ids.copy()

            pose = self._sample_free_object_pose(
                obj,
                x_range=x_range,
                y_range=y_range,
                z=0.025,
                collision_check_ids=collision_check_ids,
            )

            if is_target:
                target_pos = pose.position

            object_ids.append(obj)
            colors.append(color)

        return object_ids, colors, target_idx

    def _get_random_color(self) -> tuple[float, float, float, float]:
        colors = [
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0, 1.0),
            (0.5, 0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0, 1.0),
        ]
        return colors[int(np.random.randint(0, len(colors)))]

    def _euler_to_quat(
        self, roll: float, pitch: float, yaw: float
    ) -> tuple[float, float, float, float]:
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return (qx, qy, qz, qw)

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
            theta = np.random.uniform(0, 2 * np.pi)
            quat = self._euler_to_quat(0, 0, theta)

            pose = Pose(position=(x, y, z), orientation=quat)
            set_pose(object_id, pose, self.physics_client_id)

            p.performCollisionDetection(physicsClientId=self.physics_client_id)

            collision_free = True
            for collision_id in collision_check_ids:
                if check_body_collisions(
                    object_id,
                    collision_id,
                    self.physics_client_id,
                    distance_threshold=1e-3,
                ):
                    collision_free = False
                    break

            if collision_free:
                return pose

        raise RuntimeError(
            f"Could not sample free object position after {max_attempts} attempts. "
            f"x_range: {x_range}, y_range: {y_range}"
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        if self.robot is not None:
            p.removeBody(self.robot.robot_id, physicsClientId=self.physics_client_id)
            if self.scene is not None:
                p.removeBody(
                    self.scene.table_id, physicsClientId=self.physics_client_id
                )
                for obj_id in self.scene.object_ids:
                    p.removeBody(obj_id, physicsClientId=self.physics_client_id)

        self.robot = cast(
            FingeredSingleArmPyBulletRobot,
            create_pybullet_robot("panda", self.physics_client_id),
        )
        gripper_pos = self.robot.get_joint_positions()[7:]
        self.robot.set_joints(self.home_joint_positions + list(gripper_pos))

        self._held_object_id = None
        self._grasp_transform = None

        table_id = self._create_table()
        object_ids, colors, target_idx = self._spawn_objects_with_occlusion()

        self.scene = TabletopScene(
            robot=self.robot,
            table_id=table_id,
            object_ids=object_ids,
            object_colors=colors,
            target_idx=target_idx,
            physics_client_id=self.physics_client_id,
        )

        obs = self._get_observation()
        info = {"target_object_id": object_ids[int(target_idx)]}
        print(f"Reset environment with target object ID: {info['target_object_id']}")

        camera_image = self.get_camera_image()
        self.belief = create_initial_belief(self, camera_image, num_particles=100)

        if self.gui:
            self._update_camera_visualization()

        return obs, info

    @property
    def camera_to_hand_transform(self) -> Pose:
        """Get transform from panda_hand link to wrist camera."""
        return Pose(
            position=self.wrist_camera_offset,
            orientation=self.wrist_camera_quat,
        )

    @property
    def camera_to_ee_transform(self) -> Pose:
        """Alias for camera_to_hand_transform (for backward compatibility)."""
        return self.camera_to_hand_transform

    def _get_hand_pose(self) -> Pose:
        """Get pose of panda_hand link (where camera is mounted)."""
        assert self.robot is not None
        hand_link_id = self.robot.link_from_name("panda_hand")
        return get_link_pose(self.robot.robot_id, hand_link_id, self.physics_client_id)

    def get_camera_pose_se3(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Get camera pose as (position, orientation) tuples."""
        assert self.robot is not None

        hand_pose = self._get_hand_pose()
        camera_pose = multiply_poses(hand_pose, self.camera_to_hand_transform)

        return (camera_pose.position, camera_pose.orientation)

    def _get_observation(self) -> dict[str, np.ndarray]:
        assert self.robot is not None
        assert self.scene is not None

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
        for i, obj_id in enumerate(self.scene.object_ids):
            pose = get_pose(obj_id, self.physics_client_id)
            object_poses[i, :3] = pose.position
            object_poses[i, 3:] = pose.orientation

        held_idx = -1
        if self._held_object_id is not None:
            held_idx = self.scene.object_ids.index(self._held_object_id)

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

    def get_camera_image(self) -> CameraImage:
        """Get RGB-D-Seg image from wrist-mounted camera."""
        assert self.robot is not None

        hand_pose = self._get_hand_pose()
        camera_pose = multiply_poses(hand_pose, self.camera_to_hand_transform)

        far = 5.0
        camera_z_axis = np.array([0, 0, far])
        rot_matrix = p.getMatrixFromQuaternion(camera_pose.orientation)
        rot_matrix = np.array(rot_matrix).reshape((3, 3))
        target_point = np.array(camera_pose.position) + rot_matrix @ camera_z_axis

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pose.position,
            cameraTargetPosition=tuple(target_point),
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.physics_client_id,
        )

        fov_y = 2 * np.arctan(self.camera_height / (2 * self.camera_intrinsics.fy))
        fov_y_deg = np.degrees(fov_y)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov_y_deg,
            aspect=float(self.camera_width / self.camera_height),
            nearVal=0.02,
            farVal=5.0,
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

        far, near = 5.0, 0.02
        depth_array = far * near / (far - (far - near) * depth_array)

        return CameraImage(
            rgb=rgb_array.astype(np.uint8),
            depth=depth_array,
            segmentation=seg_array,
        )

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        assert self.robot is not None
        assert self.scene is not None

        joint_delta = action[:7]
        gripper_action = action[7] if len(action) > 7 else 0.0

        if gripper_action > 0.5 and self._held_object_id is not None:
            self._held_object_id = None
            self._grasp_transform = None

        if gripper_action < -0.5 and self._held_object_id is None:
            self._try_grasp()

        current_joints = self.robot.get_joint_positions()

        new_joints = np.array(current_joints[:7]) + joint_delta
        lower, upper = (
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        )
        new_joints = np.clip(new_joints, lower, upper)

        full_joints = list(new_joints) + list(current_joints[7:])
        self.robot.set_joints(full_joints)

        if self._held_object_id is not None and self._grasp_transform is not None:
            self._update_held_object_pose()

        p.stepSimulation(physicsClientId=self.physics_client_id)

        if self.gui:
            self._update_camera_visualization()

        if self.belief is not None:
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
                self.scene.object_ids,
                self.physics_client_id,
            )

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        return obs, reward, terminated, truncated, info

    def _update_camera_visualization(self) -> None:
        """Update wrist camera frame visualization in GUI."""
        if not self.gui or self.robot is None:
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

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            camera_image = self.get_camera_image()
            return camera_image.rgb
        if self.render_mode == "rgb_array_external":
            return capture_image(
                self.physics_client_id,
                camera_distance=1.5,
                camera_yaw=50,
                camera_pitch=-35,
                camera_target=(0.5, 0.0, 0.0),
                image_width=640,
                image_height=480,
            )
        return None

    def _try_grasp(self) -> None:
        """Try to grasp an object near the end effector.

        Uses tight distance threshold since kinematic planner should
        have positioned the EE at the exact grasp pose.
        """
        assert self.robot is not None
        assert self.scene is not None

        ee_pose = self.robot.get_end_effector_pose()

        for obj_id in self.scene.object_ids:
            obj_pose = get_pose(obj_id, self.physics_client_id)
            dist_sq = (
                (ee_pose.position[0] - obj_pose.position[0]) ** 2
                + (ee_pose.position[1] - obj_pose.position[1]) ** 2
                + (ee_pose.position[2] - obj_pose.position[2]) ** 2
            )
            if dist_sq < 1e-4:
                self._held_object_id = obj_id
                self._grasp_transform = multiply_poses(ee_pose.invert(), obj_pose)
                break

    def _update_held_object_pose(self) -> None:
        """Update held object pose based on current EE pose and grasp
        transform."""
        assert self.robot is not None
        assert self._held_object_id is not None
        assert self._grasp_transform is not None

        ee_pose = self.robot.get_end_effector_pose()
        new_obj_pose = multiply_poses(ee_pose, self._grasp_transform)
        set_pose(self._held_object_id, new_obj_pose, self.physics_client_id)

    @property
    def held_object_id(self) -> int | None:
        """Get the ID of the currently held object, if any."""
        return self._held_object_id

    def get_state(self) -> TabletopEnvState:
        """Get current env state for sim synchronization."""
        assert self.robot is not None
        assert self.scene is not None

        robot_joints = tuple(self.robot.get_joint_positions())
        object_poses = tuple(
            get_pose(obj_id, self.physics_client_id) for obj_id in self.scene.object_ids
        )

        held_object_idx = None
        if self._held_object_id is not None:
            held_object_idx = self.scene.object_ids.index(self._held_object_id)

        return TabletopEnvState(
            robot_joints=robot_joints,
            object_poses=object_poses,
            held_object_idx=held_object_idx,
            grasp_transform=self._grasp_transform,
        )

    def set_state(self, state: TabletopEnvState) -> None:
        """Set env state from TabletopEnvState."""
        assert self.robot is not None
        assert self.scene is not None

        joints = list(state.robot_joints)
        # Ensure finger symmetry (pybullet_helpers requires this)
        if len(joints) >= 9:
            finger_avg = (joints[7] + joints[8]) / 2
            joints[7] = finger_avg
            joints[8] = finger_avg
        self.robot.set_joints(joints)

        for i, pose in enumerate(state.object_poses):
            obj_id = self.scene.object_ids[i]
            set_pose(obj_id, pose, self.physics_client_id)

        if state.held_object_idx is not None:
            self._held_object_id = self.scene.object_ids[state.held_object_idx]
            self._grasp_transform = state.grasp_transform
        else:
            self._held_object_id = None
            self._grasp_transform = None

    def get_collision_ids(self) -> set[int]:
        """Get IDs for collision checking during motion planning."""
        if self.scene is None:
            return set()
        ids = {self.scene.table_id}
        ids.update(self.scene.object_ids)
        return ids

    def close(self):
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
