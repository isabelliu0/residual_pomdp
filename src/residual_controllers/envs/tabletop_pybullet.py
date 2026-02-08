"""PyBullet environment for tabletop manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.gui import create_gui_connection
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


class TabletopPickEnv(gym.Env):
    """Tabletop manipulation: robot must pick a target object from a cluttered table."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        gui: bool = False,
        num_objects: int = 5,
        occlusion_prob: float = 0.7,
        camera_width: int = 640,
        camera_height: int = 480,
    ):
        super().__init__()

        self.gui = gui
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        self.camera_width = camera_width
        self.camera_height = camera_height

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

        fov = 60
        self.camera_intrinsics = CameraIntrinsics(
            fx=camera_width / (2 * np.tan(np.radians(fov) / 2)),
            fy=camera_height / (2 * np.tan(np.radians(fov) / 2)),
            cx=camera_width / 2.0,
            cy=camera_height / 2.0,
            width=camera_width,
            height=camera_height,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "joint_positions": gym.spaces.Box(
                    low=-np.pi, high=np.pi, shape=(9,), dtype=np.float32
                ),
                "camera_pose": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(7,), dtype=np.float32
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

        camera_image = self.get_camera_image()
        self.belief = create_initial_belief(self, camera_image, num_particles=100)

        return obs, info

    def get_camera_pose_se3(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Get camera pose as (position, orientation) tuples."""
        assert self.robot is not None
        camera_link_id = self.robot.end_effector_id
        camera_pose = get_link_pose(
            self.robot.robot_id,
            camera_link_id,
            self.physics_client_id,
        )
        return (camera_pose.position, camera_pose.orientation)

    def _get_observation(self) -> dict[str, np.ndarray]:
        assert self.robot is not None

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

        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "camera_pose": camera_pose_array.astype(np.float32),
        }

    def get_camera_image(self) -> CameraImage:
        """Get the current camera image from the robot's end-effector view."""
        assert self.robot is not None

        camera_link_id = self.robot.end_effector_id
        camera_pose = get_link_pose(
            self.robot.robot_id,
            camera_link_id,
            self.physics_client_id,
        )

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_pose.position,
            distance=0.5,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client_id,
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.camera_width / self.camera_height),
            nearVal=0.01,
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

        far, near = 5.0, 0.01
        depth_array = far * near / (far - (far - near) * depth_array)

        return CameraImage(
            rgb=rgb_array.astype(np.uint8),
            depth=depth_array,
            segmentation=seg_array,
        )

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        assert self.robot is not None
        assert self.scene is not None

        current_joints = self.robot.get_joint_positions()

        new_joints = np.array(current_joints[:7]) + action
        lower, upper = (
            self.robot.joint_lower_limits[:7],
            self.robot.joint_upper_limits[:7],
        )
        new_joints = np.clip(new_joints, lower, upper)

        full_joints = list(new_joints) + current_joints[7:]
        self.robot.set_joints(full_joints)

        p.stepSimulation(physicsClientId=self.physics_client_id)

        if self.belief is not None:
            self.belief = predict_belief(
                self.belief,
                action,
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

    def render(self):
        """Render the environment."""
        if self.gui:
            return None
        camera_image = self.get_camera_image()
        return camera_image.rgb

    def close(self):
        if self.physics_client_id is not None:
            p.disconnect(physicsClientId=self.physics_client_id)
