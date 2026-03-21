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
from tomsgeoms2d.structs import Circle

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
    target_area_id: int
    physics_client_id: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


@dataclass(frozen=True)
class TabletopEnvState:
    """State for env/sim synchronization."""

    robot_joints: tuple[float, ...]
    object_poses: tuple[Pose, ...]
    held_object_idx: int | None
    grasp_transform: Pose | None
    target_area_pose: Pose | None = None


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

        self.robot: FingeredSingleArmPyBulletRobot = cast(
            FingeredSingleArmPyBulletRobot,
            create_pybullet_robot("panda", self.physics_client_id),
        )
        gripper_pos = self.robot.get_joint_positions()[7:]
        self.robot.set_joints(self.home_joint_positions + list(gripper_pos))

        self.scene: TabletopScene | None = None
        self.belief: Belief | None = None
        self._camera_frame_ids: set[int] = set()
        self._held_object_id: int | None = None
        self._grasp_transform: Pose | None = None

        self._table_id = self._create_table()

        _distractor_colors = [
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0, 1.0),
            (0.5, 0.0, 1.0, 1.0),
        ]
        _target_color = (1.0, 0.0, 0.0, 1.0)
        self._object_ids: list[int] = []
        self._object_colors: list[tuple[float, float, float, float]] = []

        target_block = create_pybullet_block(
            color=_target_color,
            half_extents=(0.025, 0.025, 0.025),
            physics_client_id=self.physics_client_id,
            mass=0.1,
            friction=0.9,
        )
        set_pose(target_block, Pose(position=(-10.0, 0.0, 0.0)), self.physics_client_id)
        self._object_ids.append(target_block)
        self._object_colors.append(_target_color)

        for i in range(num_objects - 1):
            color = _distractor_colors[i % len(_distractor_colors)]
            obj = create_pybullet_block(
                color=color,
                half_extents=(0.025, 0.025, 0.025),
                physics_client_id=self.physics_client_id,
                mass=0.1,
                friction=0.9,
            )
            set_pose(
                obj, Pose(position=(-10.0, float(i + 1), 0.0)), self.physics_client_id
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
            Pose(position=(-10.0, float(num_objects), 0.0)),
            self.physics_client_id,
        )

        self._object_labels = [chr(65 + i) for i in range(num_objects)]
        self._label_to_id: dict[str, int] = dict(
            zip(self._object_labels, self._object_ids)
        )
        self._id_to_label: dict[int, str] = {v: k for k, v in self._label_to_id.items()}

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
                "target_area_pose": gym.spaces.Box(
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

            yaw = np.random.uniform(0, 2 * np.pi)
            quat = self._euler_to_quat(0, 0, yaw)
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
            f"Could not sample free target position after {max_attempts} attempts. "
            f"r_min: {r_min}, r_max: {r_max}"
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

        gripper_pos = self.robot.get_joint_positions()[7:]
        self.robot.set_joints(self.home_joint_positions + list(gripper_pos))

        self._held_object_id = None
        self._grasp_transform = None

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

        self.scene = TabletopScene(
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

        obs = self.get_observation()
        info = {"target_object_label": "A"}

        camera_image = self.get_camera_image()
        self.belief = create_initial_belief(self, camera_image, num_particles=100)

        if self.gui:
            self._update_camera_visualization()

        return obs, info

    @property
    def object_labels(self) -> list[str]:
        """Stable object labels for this environment."""
        return self._object_labels

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

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation."""
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

        target_area_pose_arr = np.zeros(7, dtype=np.float32)
        if self.scene is not None:
            area_pose = get_pose(self.scene.target_area_id, self.physics_client_id)
            target_area_pose_arr[:3] = area_pose.position
            target_area_pose_arr[3:] = area_pose.orientation

        return {
            "joint_positions": np.array(joint_positions, dtype=np.float32),
            "camera_pose": camera_pose_array.astype(np.float32),
            "object_poses": object_poses,
            "held_object_idx": np.array([held_idx], dtype=np.int32),
            "grasp_transform": grasp_tf,
            "target_area_pose": target_area_pose_arr,
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

        camera_up = tuple(-rot_matrix[:, 1])
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pose.position,
            cameraTargetPosition=tuple(target_point),
            cameraUpVector=camera_up,
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
            held_label = (
                self.scene.id_to_label.get(self._held_object_id)
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
            area_aabb = p.getAABB(
                self.scene.target_area_id, physicsClientId=self.physics_client_id
            )
            excluded_aabbs = [
                (
                    float(area_aabb[0][0]),
                    float(area_aabb[0][1]),
                    float(area_aabb[1][0]),
                    float(area_aabb[1][1]),
                )
            ]
            self.belief = update_belief(
                self.belief,
                camera_image,
                camera_pose,
                self.camera_intrinsics,
                self.scene.label_to_id,
                self.physics_client_id,
                table_id=self.scene.table_id,
                excluded_aabbs=excluded_aabbs,
            )

        obs = self.get_observation()

        terminated = self._held_object_id is None and self._is_target_in_area()
        reward = 1.0 if terminated else 0.0
        truncated = False
        info = {"target_object_label": "A"}

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

    def _is_target_in_area(self) -> bool:
        """Check if the target block is fully contained within the target
        area."""
        assert self.scene is not None
        target_id = self.scene.object_ids[self.scene.target_idx]
        target_aabb = p.getAABB(target_id, physicsClientId=self.physics_client_id)
        area_aabb = p.getAABB(
            self.scene.target_area_id, physicsClientId=self.physics_client_id
        )
        return (
            target_aabb[0][0] >= area_aabb[0][0]
            and target_aabb[1][0] <= area_aabb[1][0]
            and target_aabb[0][1] >= area_aabb[0][1]
            and target_aabb[1][1] <= area_aabb[1][1]
        )

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

    @held_object_id.setter
    def held_object_id(self, value: int | None) -> None:
        self._held_object_id = value

    @property
    def grasp_transform(self) -> Pose | None:
        """Get the current grasp transform (EE-to-object), if any."""
        return self._grasp_transform

    @grasp_transform.setter
    def grasp_transform(self, value: Pose | None) -> None:
        self._grasp_transform = value

    def get_state(self) -> TabletopEnvState:
        """Get current env state for sim synchronization."""
        assert self.robot is not None
        assert self.scene is not None

        robot_joints = tuple(self.robot.get_joint_positions())
        object_poses = tuple(
            get_pose(obj_id, self.physics_client_id) for obj_id in self.scene.object_ids
        )
        target_area_pose = get_pose(self.scene.target_area_id, self.physics_client_id)

        held_object_idx = None
        if self._held_object_id is not None:
            held_object_idx = self.scene.object_ids.index(self._held_object_id)

        return TabletopEnvState(
            robot_joints=robot_joints,
            object_poses=object_poses,
            held_object_idx=held_object_idx,
            grasp_transform=self._grasp_transform,
            target_area_pose=target_area_pose,
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

        if state.target_area_pose is not None:
            set_pose(
                self.scene.target_area_id,
                state.target_area_pose,
                self.physics_client_id,
            )

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
