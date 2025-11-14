"""Cover2D environment for residual RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from tomsgeoms2d.structs import Circle, Rectangle


class GripperAction(Enum):
    """Gripper actions."""

    NOOP = 0
    PICK = 1
    PLACE = 2


@dataclass
class Pose2D:
    """2D pose with x, y, and theta."""

    x: float
    y: float
    theta: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert pose to numpy array."""
        return np.array([self.x, self.y, self.theta], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> Pose2D:
        """Create Pose2D from numpy array."""
        return Pose2D(x=float(arr[0]), y=float(arr[1]), theta=float(arr[2]))

    def __hash__(self):
        return hash((self.x, self.y, self.theta))


@dataclass
class GripperState:
    """Gripper state indicating if holding an object."""

    is_holding: bool
    held_object_id: int | None = None

    def __hash__(self):
        return hash((self.is_holding, self.held_object_id))


@dataclass
class ObjectPose:
    """Pose of an object in the environment."""

    object_id: int
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        """Convert object pose to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)

    def __hash__(self):
        return hash((self.object_id, self.x, self.y))


@dataclass
class State:
    """Full state of the environment."""

    robot_pose: Pose2D
    gripper_state: GripperState
    object_poses: dict[int, ObjectPose]

    def __hash__(self):
        return hash(
            (self.robot_pose, self.gripper_state, tuple(self.object_poses.items()))
        )


@dataclass
class Observation:
    """Observation of the environment."""

    robot_pose: Pose2D | None
    gripper_state: GripperState


@dataclass
class Action:
    """Action in the Cover2D environment."""

    dx: float
    dy: float
    dtheta: float = 0.0
    gripper_action: GripperAction = GripperAction.NOOP

    def to_array(self) -> np.ndarray:
        """Convert action to numpy array."""
        return np.array(
            [self.dx, self.dy, self.dtheta, self.gripper_action.value], dtype=np.float32
        )

    @staticmethod
    def from_array(arr: np.ndarray) -> Action:
        """Create Action from numpy array."""
        return Action(
            dx=float(arr[0]),
            dy=float(arr[1]),
            dtheta=float(arr[2]),
            gripper_action=GripperAction(int(arr[3])),
        )


@dataclass
class Cover2DConfig:
    """Configuration for Cover2D environment.

    Three regions:
    - Region 1: Observable, no transition noise
    - Region 2: No observations, high transition noise
    - Region 3: No observations, no transition noise
    """

    world_width: float = 10.0
    world_height: float = 6.0
    region1_end_x: float = 2.5
    region2_end_x: float = 7.0
    goal_region_x: float = 9.2
    goal_region_y: float = 2.5
    goal_region_width: float = 0.8
    goal_region_height: float = 0.8
    robot_radius: float = 0.3
    block_size: float = 0.3
    transition_noise_std: float = 0.3
    num_particles: int = 10
    initial_robot_x: float = 1.5
    initial_robot_y: float = 3.5
    initial_robot_theta: float = 0.0
    initial_block_x: float = 1.5
    initial_block_y: float = 2.0
    seed: int = 0


@dataclass
class Belief:
    """Belief state represented by particles and weights."""

    particles: list[State]
    weights: np.ndarray

    def __post_init__(self):
        assert len(self.particles) == len(self.weights)
        self.weights = self.weights / np.sum(self.weights)

    @property
    def num_particles(self) -> int:
        """Get number of particles in the belief."""
        return len(self.particles)


class World:
    """World model for Cover2D environment."""

    def __init__(self, config: Cover2DConfig):
        self.config = config
        self.robot_radius = config.robot_radius
        self.block_size = config.block_size

        self.goal_region = Rectangle(
            x=config.goal_region_x,
            y=config.goal_region_y,
            width=config.goal_region_width,
            height=config.goal_region_height,
            theta=0.0,
        )
        self.region2 = Rectangle(
            x=config.region1_end_x,
            y=0.0,
            width=config.region2_end_x - config.region1_end_x,
            height=config.world_height,
            theta=0.0,
        )
        self.region3 = Rectangle(
            x=config.region2_end_x,
            y=0.0,
            width=config.world_width - config.region2_end_x,
            height=config.world_height,
            theta=0.0,
        )

    def get_robot_shape(self, x: float, y: float) -> Circle:
        """Get robot shape as a circle."""
        return Circle(x=x, y=y, radius=self.robot_radius)

    def get_block_shape(self, x: float, y: float, theta: float = 0.0) -> Rectangle:
        """Get block shape as a rectangle."""
        return Rectangle(
            x=x,
            y=y,
            width=self.block_size,
            height=self.block_size,
            theta=theta,
        )

    def get_gripper_center(
        self, robot_x: float, robot_y: float, theta: float
    ) -> tuple[float, float]:
        """Get gripper center position based on robot pose."""
        finger_width = self.robot_radius * 1.0
        gripper_offset = self.robot_radius + finger_width / 2
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        gripper_x = robot_x + gripper_offset * cos_theta
        gripper_y = robot_y + gripper_offset * sin_theta
        return gripper_x, gripper_y

    def get_held_object_position(
        self, robot_x: float, robot_y: float, theta: float
    ) -> tuple[float, float]:
        """Get held object position based on robot pose."""
        finger_width = self.robot_radius * 1.0
        object_offset = self.robot_radius + finger_width * 0.8
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        obj_x = robot_x + object_offset * cos_theta
        obj_y = robot_y + object_offset * sin_theta
        return obj_x, obj_y

    def get_gripper_shapes(
        self, x: float, y: float, theta: float = 0.0
    ) -> list[Rectangle]:
        """Get gripper shapes as rectangles."""
        finger_width = self.robot_radius * 1.0
        finger_height = self.robot_radius * 0.25
        finger_spacing = self.robot_radius * 1.5
        connector_width = finger_spacing
        connector_height = self.robot_radius * 0.25

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        finger_forward_offset = self.robot_radius + finger_width / 2
        connector_forward_offset = self.robot_radius + connector_height / 2

        left_finger_x = (
            x + finger_forward_offset * cos_theta - finger_spacing / 2 * sin_theta
        )
        left_finger_y = (
            y + finger_forward_offset * sin_theta + finger_spacing / 2 * cos_theta
        )

        right_finger_x = (
            x + finger_forward_offset * cos_theta + finger_spacing / 2 * sin_theta
        )
        right_finger_y = (
            y + finger_forward_offset * sin_theta - finger_spacing / 2 * cos_theta
        )

        connector_x = x + connector_forward_offset * cos_theta
        connector_y = y + connector_forward_offset * sin_theta

        return [
            Rectangle(
                x=left_finger_x,
                y=left_finger_y,
                width=finger_width,
                height=finger_height,
                theta=theta,
            ),
            Rectangle(
                x=right_finger_x,
                y=right_finger_y,
                width=finger_width,
                height=finger_height,
                theta=theta,
            ),
            Rectangle(
                x=connector_x,
                y=connector_y,
                width=connector_height,
                height=connector_width,
                theta=theta,
            ),
        ]

    def in_bounds(self, x: float, y: float) -> bool:
        """Check if (x, y) is within world bounds."""
        return 0 <= x <= self.config.world_width and 0 <= y <= self.config.world_height

    def in_region1(self, x: float) -> bool:
        """Check if x is in region 1."""
        return x < self.config.region1_end_x

    def in_region2(self, x: float) -> bool:
        """Check if x is in region 2."""
        return self.config.region1_end_x <= x < self.config.region2_end_x

    def in_region3(self, x: float) -> bool:
        """Check if x is in region 3."""
        return x >= self.config.region2_end_x

    def has_observation(self, x: float) -> bool:
        """Check if robot at x gets observations in the corresponding
        region."""
        return self.in_region1(x)

    def in_goal_region(self, x: float, y: float) -> bool:
        """Check if block centered at (x, y) is entirely in the goal region."""
        half_size = self.block_size / 2
        return (
            self.config.goal_region_x <= x - half_size
            and x + half_size
            <= self.config.goal_region_x + self.config.goal_region_width
            and self.config.goal_region_y <= y - half_size
            and y + half_size
            <= self.config.goal_region_y + self.config.goal_region_height
        )

    def check_robot_collision(self, x: float, y: float) -> bool:
        """Check if robot at (x, y) collides with world boundaries."""
        if not self.in_bounds(x, y):
            return True
        if x - self.robot_radius < 0 or x + self.robot_radius > self.config.world_width:
            return True
        if (
            y - self.robot_radius < 0
            or y + self.robot_radius > self.config.world_height
        ):
            return True
        return False

    def check_block_collision(self, block_x: float, block_y: float) -> bool:
        """Check if block at (block_x, block_y) collides with world
        boundaries."""
        half_size = self.block_size / 2
        if block_x - half_size < 0 or block_x + half_size > self.config.world_width:
            return True
        if block_y - half_size < 0 or block_y + half_size > self.config.world_height:
            return True
        return False

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def can_grasp(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        block_x: float,
        block_y: float,
    ) -> bool:
        """Check if the robot can grasp the block at given positions."""
        gripper_x, gripper_y = self.get_gripper_center(robot_x, robot_y, robot_theta)
        dist = self.distance(gripper_x, gripper_y, block_x, block_y)
        grasp_threshold = self.block_size / 2 + 0.2  # to be adjusted
        return dist <= grasp_threshold


def apply_action(state: State, action: Action, world: World) -> State:
    """Apply action to state and return new state."""
    new_x = state.robot_pose.x + action.dx
    new_y = state.robot_pose.y + action.dy
    new_theta = state.robot_pose.theta + action.dtheta
    new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

    finger_width = world.robot_radius * 1.0
    if state.gripper_state.is_holding:
        object_offset = world.robot_radius + finger_width * 0.8
        max_extent = object_offset + world.block_size / 2
    else:
        max_extent = world.robot_radius + finger_width

    min_x = max_extent
    max_x = world.config.world_width - max_extent
    min_y = max_extent
    max_y = world.config.world_height - max_extent

    new_x = np.clip(new_x, min_x, max_x)
    new_y = np.clip(new_y, min_y, max_y)

    new_robot_pose = Pose2D(x=new_x, y=new_y, theta=new_theta)
    new_gripper_state = state.gripper_state
    new_object_poses = dict(state.object_poses)

    if action.gripper_action == GripperAction.PICK:
        if not state.gripper_state.is_holding:
            for obj_id, obj_pose in state.object_poses.items():
                if world.can_grasp(new_x, new_y, new_theta, obj_pose.x, obj_pose.y):
                    new_gripper_state = GripperState(
                        is_holding=True, held_object_id=obj_id
                    )
                    break

    elif action.gripper_action == GripperAction.PLACE:
        if state.gripper_state.is_holding:
            new_gripper_state = GripperState(is_holding=False, held_object_id=None)

    if new_gripper_state.is_holding and new_gripper_state.held_object_id is not None:
        obj_id = new_gripper_state.held_object_id
        obj_x, obj_y = world.get_held_object_position(new_x, new_y, new_theta)
        new_object_poses[obj_id] = ObjectPose(object_id=obj_id, x=obj_x, y=obj_y)

    return State(
        robot_pose=new_robot_pose,
        gripper_state=new_gripper_state,
        object_poses=new_object_poses,
    )


def sample_next_state(
    state: State, action: Action, world: World, rng: np.random.Generator
) -> State:
    """Sample next state given current state and action with noise."""
    if world.in_region2(state.robot_pose.x):
        noise_x = rng.normal(0, world.config.transition_noise_std)
        noise_y = rng.normal(0, world.config.transition_noise_std)
        noise_theta = rng.normal(0, world.config.transition_noise_std * 0.5)
    else:
        noise_x = 0.0
        noise_y = 0.0
        noise_theta = 0.0

    noisy_action = Action(
        dx=action.dx + noise_x,
        dy=action.dy + noise_y,
        dtheta=action.dtheta + noise_theta,
        gripper_action=action.gripper_action,
    )

    return apply_action(state, noisy_action, world)


def get_observation(state: State, world: World) -> Observation:
    """Get observation from state."""
    if not world.has_observation(state.robot_pose.x):
        return Observation(robot_pose=None, gripper_state=state.gripper_state)

    observed_pose = Pose2D(
        x=state.robot_pose.x,
        y=state.robot_pose.y,
        theta=state.robot_pose.theta,
    )

    return Observation(robot_pose=observed_pose, gripper_state=state.gripper_state)


def create_initial_belief(
    initial_state: State, config: Cover2DConfig, rng: np.random.Generator
) -> Belief:
    """Create initial belief with particles around the initial state."""
    particles = [initial_state]

    initial_noise_std = 0.05
    for _ in range(config.num_particles - 1):
        noise_x = rng.normal(0, initial_noise_std)
        noise_y = rng.normal(0, initial_noise_std)
        noise_theta = rng.normal(0, initial_noise_std)

        noisy_pose = Pose2D(
            x=initial_state.robot_pose.x + noise_x,
            y=initial_state.robot_pose.y + noise_y,
            theta=initial_state.robot_pose.theta + noise_theta,
        )

        particle = State(
            robot_pose=noisy_pose,
            gripper_state=initial_state.gripper_state,
            object_poses=dict(initial_state.object_poses),
        )
        particles.append(particle)

    weights = np.ones(config.num_particles) / config.num_particles
    return Belief(particles=particles, weights=weights)


def predict_belief(
    belief: Belief, action: Action, world: World, rng: np.random.Generator
) -> Belief:
    """Predict next belief given current belief and action."""
    predicted_particles = []
    for particle in belief.particles:
        next_particle = sample_next_state(particle, action, world, rng)
        predicted_particles.append(next_particle)

    return Belief(particles=predicted_particles, weights=belief.weights.copy())


def observation_likelihood(
    state: State, observation: Observation, world: World
) -> float:
    """Compute likelihood of observation given state."""
    if observation.robot_pose is None:
        if world.has_observation(state.robot_pose.x):
            return 0.01
        return 1.0
    if not world.has_observation(state.robot_pose.x):
        return 0.01

    dx = observation.robot_pose.x - state.robot_pose.x
    dy = observation.robot_pose.y - state.robot_pose.y
    dtheta = observation.robot_pose.theta - state.robot_pose.theta

    std = 0.001
    likelihood = np.exp(
        -0.5 * ((dx / std) ** 2 + (dy / std) ** 2 + (dtheta / std) ** 2)
    )

    if state.gripper_state.is_holding != observation.gripper_state.is_holding:
        likelihood *= 0.01

    return likelihood


def update_belief(
    predicted_belief: Belief, observation: Observation, world: World
) -> Belief:
    """Update belief with observation using particle weights."""
    new_weights = np.array(
        [
            predicted_belief.weights[i]
            * observation_likelihood(particle, observation, world)
            for i, particle in enumerate(predicted_belief.particles)
        ]
    )

    if np.sum(new_weights) < 1e-10:
        new_weights = np.ones(len(new_weights)) / len(new_weights)
    else:
        new_weights = new_weights / np.sum(new_weights)

    effective_sample_size = 1.0 / np.sum(new_weights**2)
    if (
        effective_sample_size < len(predicted_belief.particles) / 2
    ):  # too many low-weight particles
        return resample_belief(predicted_belief.particles, new_weights)

    return Belief(particles=predicted_belief.particles, weights=new_weights)


def resample_belief(particles: list[State], weights: np.ndarray) -> Belief:
    """Resample belief particles based on weights."""
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    resampled_particles = [particles[i] for i in indices]
    uniform_weights = np.ones(len(particles)) / len(particles)
    return Belief(particles=resampled_particles, weights=uniform_weights)


def get_mean_state(belief: Belief) -> State:
    """Compute mean state from belief particles and weights.

    Uses weighted average for robot pose and object poses. Uses best
    particle for discrete gripper state.
    """
    x_mean = np.average(
        [p.robot_pose.x for p in belief.particles], weights=belief.weights
    )
    y_mean = np.average(
        [p.robot_pose.y for p in belief.particles], weights=belief.weights
    )
    theta_mean = np.average(
        [p.robot_pose.theta for p in belief.particles], weights=belief.weights
    )

    mean_robot_pose = Pose2D(x=float(x_mean), y=float(y_mean), theta=float(theta_mean))

    best_particle_idx = np.argmax(belief.weights)
    best_particle = belief.particles[best_particle_idx]

    mean_object_poses = {}
    if belief.particles[0].object_poses:
        for obj_id in belief.particles[0].object_poses.keys():
            obj_x_mean = np.average(
                [
                    p.object_poses[obj_id].x
                    for p in belief.particles
                    if obj_id in p.object_poses
                ],
                weights=belief.weights,
            )
            obj_y_mean = np.average(
                [
                    p.object_poses[obj_id].y
                    for p in belief.particles
                    if obj_id in p.object_poses
                ],
                weights=belief.weights,
            )
            mean_object_poses[obj_id] = ObjectPose(
                object_id=obj_id, x=float(obj_x_mean), y=float(obj_y_mean)
            )

    return State(
        robot_pose=mean_robot_pose,
        gripper_state=best_particle.gripper_state,
        object_poses=mean_object_poses,
    )


class PickController:
    """Simple pick controller for Cover2D environment."""

    def __init__(self, world: World, target_object_id: int):
        self.world = world
        self.target_object_id = target_object_id

    def get_action(self, belief: Belief) -> Action:
        """Get action to pick the target object."""
        mean_state = get_mean_state(belief)

        if mean_state.gripper_state.is_holding:
            return Action(dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.NOOP)

        if self.target_object_id not in mean_state.object_poses:
            return Action(dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.NOOP)

        target_obj = mean_state.object_poses[self.target_object_id]
        robot_x = mean_state.robot_pose.x
        robot_y = mean_state.robot_pose.y
        robot_theta = mean_state.robot_pose.theta

        if self.world.can_grasp(
            robot_x, robot_y, robot_theta, target_obj.x, target_obj.y
        ):
            return Action(dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.PICK)

        gripper_x, gripper_y = self.world.get_gripper_center(
            robot_x, robot_y, robot_theta
        )
        dx_to_obj = target_obj.x - gripper_x
        dy_to_obj = target_obj.y - gripper_y
        dist = np.sqrt(dx_to_obj**2 + dy_to_obj**2)

        step_size = 0.5
        if dist > 0:
            dx = (dx_to_obj / dist) * min(step_size, dist)
            dy = (dy_to_obj / dist) * min(step_size, dist)
        else:
            dx, dy = 0.0, 0.0

        target_theta = np.arctan2(dy_to_obj, dx_to_obj)
        dtheta = target_theta - robot_theta
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        return Action(
            dx=dx, dy=dy, dtheta=dtheta * 0.5, gripper_action=GripperAction.NOOP
        )


class PlaceController:
    """Simple place controller for Cover2D environment."""

    def __init__(self, world: World, target_x: float, target_y: float):
        self.world = world
        self.target_x = target_x
        self.target_y = target_y

    def get_action(self, belief: Belief) -> Action:
        """Get action to place the held object at target location."""
        mean_state = get_mean_state(belief)

        if not mean_state.gripper_state.is_holding:
            return Action(dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.NOOP)

        robot_x = mean_state.robot_pose.x
        robot_y = mean_state.robot_pose.y
        robot_theta = mean_state.robot_pose.theta

        gripper_x, gripper_y = self.world.get_gripper_center(
            robot_x, robot_y, robot_theta
        )

        dx_to_target = self.target_x - gripper_x
        dy_to_target = self.target_y - gripper_y
        dist = np.sqrt(dx_to_target**2 + dy_to_target**2)

        if dist < 0.2:
            return Action(
                dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.PLACE
            )

        step_size = 0.5
        if dist > 0:
            dx = (dx_to_target / dist) * min(step_size, dist)
            dy = (dy_to_target / dist) * min(step_size, dist)
        else:
            dx, dy = 0.0, 0.0

        target_theta = np.arctan2(dy_to_target, dx_to_target)
        dtheta = target_theta - robot_theta
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        return Action(
            dx=dx, dy=dy, dtheta=dtheta * 0.5, gripper_action=GripperAction.NOOP
        )


class Cover2DEnv:
    """Cover2D environment class."""

    def __init__(self, config: Cover2DConfig | None = None):
        if config is None:
            config = Cover2DConfig()
        self.config = config
        self.world = World(config)
        self.rng = np.random.default_rng(config.seed)
        self.state: State | None = None
        self.belief: Belief | None = None
        self.steps = 0

    def reset(self) -> tuple[Belief, dict[str, Any]]:
        """Reset the environment to the initial state."""
        initial_robot_pose = Pose2D(
            x=self.config.initial_robot_x,
            y=self.config.initial_robot_y,
            theta=self.config.initial_robot_theta,
        )
        initial_gripper = GripperState(is_holding=False, held_object_id=None)
        initial_object_poses = {
            0: ObjectPose(
                object_id=0,
                x=self.config.initial_block_x,
                y=self.config.initial_block_y,
            )
        }

        self.state = State(
            robot_pose=initial_robot_pose,
            gripper_state=initial_gripper,
            object_poses=initial_object_poses,
        )

        self.belief = create_initial_belief(self.state, self.config, self.rng)
        self.steps = 0

        info = {"state": self.state}
        return self.belief, info

    def step(self, action: Action) -> tuple[Belief, float, bool, dict[str, Any]]:
        """Take a step in the environment with the given action."""
        assert self.state is not None
        assert self.belief is not None

        self.state = sample_next_state(self.state, action, self.world, self.rng)
        observation = get_observation(self.state, self.world)

        predicted_belief = predict_belief(self.belief, action, self.world, self.rng)
        self.belief = update_belief(predicted_belief, observation, self.world)

        reward = self._compute_reward(self.state)
        terminal = self._is_terminal(self.state)
        self.steps += 1

        info = {"state": self.state, "observation": observation}
        return self.belief, reward, terminal, info

    def _compute_reward(self, state: State) -> float:
        if (
            state.gripper_state.is_holding
            and state.gripper_state.held_object_id is not None
        ):
            obj_id = state.gripper_state.held_object_id
            if obj_id in state.object_poses:
                obj_pose = state.object_poses[obj_id]
                if self.world.in_goal_region(obj_pose.x, obj_pose.y):
                    return 100.0

        for obj_pose in state.object_poses.values():
            if self.world.in_goal_region(obj_pose.x, obj_pose.y):
                return 100.0

        return -1.0

    def _is_terminal(self, state: State) -> bool:
        for obj_pose in state.object_poses.values():
            if self.world.in_goal_region(obj_pose.x, obj_pose.y):
                return True
        return False

    def render(self, ax: Axes | None = None, show_belief: bool = True) -> Axes:
        """Render the current state of the environment."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        assert self.state is not None
        assert self.belief is not None

        ax.clear()
        ax.set_xlim(0, self.config.world_width)
        ax.set_ylim(0, self.config.world_height)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        region2_rect = patches.Rectangle(
            (self.config.region1_end_x, 0),
            self.config.region2_end_x - self.config.region1_end_x,
            self.config.world_height,
            linewidth=0,
            edgecolor="none",
            facecolor="orange",
            alpha=0.2,
        )
        ax.add_patch(region2_rect)

        region3_rect = patches.Rectangle(
            (self.config.region2_end_x, 0),
            self.config.world_width - self.config.region2_end_x,
            self.config.world_height,
            linewidth=0,
            edgecolor="none",
            facecolor="gray",
            alpha=0.2,
        )
        ax.add_patch(region3_rect)

        goal_rect = patches.Rectangle(
            (self.config.goal_region_x, self.config.goal_region_y),
            self.config.goal_region_width,
            self.config.goal_region_height,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.3,
        )
        ax.add_patch(goal_rect)

        if show_belief:
            num_particles_to_show = min(5, len(self.belief.particles))
            indices = np.linspace(
                0, len(self.belief.particles) - 1, num_particles_to_show, dtype=int
            )

            for idx in indices:
                particle = self.belief.particles[idx]

                robot_circle = patches.Circle(
                    (particle.robot_pose.x, particle.robot_pose.y),
                    self.world.robot_radius,
                    linewidth=0.5,
                    edgecolor="blue",
                    facecolor="blue",
                    alpha=0.1,
                )
                ax.add_patch(robot_circle)

                particle_gripper_shapes = self.world.get_gripper_shapes(
                    particle.robot_pose.x,
                    particle.robot_pose.y,
                    particle.robot_pose.theta,
                )
                for gripper_rect in particle_gripper_shapes:
                    cos_t = np.cos(gripper_rect.theta)
                    sin_t = np.sin(gripper_rect.theta)
                    bottom_left_x = (
                        gripper_rect.x
                        - gripper_rect.width / 2 * cos_t
                        + gripper_rect.height / 2 * sin_t
                    )
                    bottom_left_y = (
                        gripper_rect.y
                        - gripper_rect.width / 2 * sin_t
                        - gripper_rect.height / 2 * cos_t
                    )
                    rect_patch = patches.Rectangle(
                        (bottom_left_x, bottom_left_y),
                        gripper_rect.width,
                        gripper_rect.height,
                        angle=np.degrees(gripper_rect.theta),
                        linewidth=0.5,
                        edgecolor="blue",
                        facecolor="blue",
                        alpha=0.1,
                    )
                    ax.add_patch(rect_patch)

                if (
                    particle.gripper_state.is_holding
                    and particle.gripper_state.held_object_id is not None
                ):
                    obj_id = particle.gripper_state.held_object_id
                    if obj_id in particle.object_poses:
                        obj_pose = particle.object_poses[obj_id]
                        block_rect = patches.Rectangle(
                            (
                                obj_pose.x - self.world.block_size / 2,
                                obj_pose.y - self.world.block_size / 2,
                            ),
                            self.world.block_size,
                            self.world.block_size,
                            linewidth=0.5,
                            edgecolor="blue",
                            facecolor="blue",
                            alpha=0.1,
                        )
                        ax.add_patch(block_rect)

        robot_circle = patches.Circle(
            (self.state.robot_pose.x, self.state.robot_pose.y),
            self.world.robot_radius,
            linewidth=2,
            edgecolor="blue",
            facecolor="blue",
            alpha=1.0,
        )
        ax.add_patch(robot_circle)

        gripper_shapes = self.world.get_gripper_shapes(
            self.state.robot_pose.x,
            self.state.robot_pose.y,
            self.state.robot_pose.theta,
        )
        for gripper_rect in gripper_shapes:
            cos_t = np.cos(gripper_rect.theta)
            sin_t = np.sin(gripper_rect.theta)

            bottom_left_x = (
                gripper_rect.x
                - gripper_rect.width / 2 * cos_t
                + gripper_rect.height / 2 * sin_t
            )
            bottom_left_y = (
                gripper_rect.y
                - gripper_rect.width / 2 * sin_t
                - gripper_rect.height / 2 * cos_t
            )

            rect_patch = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                gripper_rect.width,
                gripper_rect.height,
                angle=np.degrees(gripper_rect.theta),
                linewidth=1,
                edgecolor="blue",
                facecolor="blue",
                alpha=1.0,
            )
            ax.add_patch(rect_patch)

        for obj_pose in self.state.object_poses.values():
            block_rect = patches.Rectangle(
                (
                    obj_pose.x - self.world.block_size / 2,
                    obj_pose.y - self.world.block_size / 2,
                ),
                self.world.block_size,
                self.world.block_size,
                linewidth=2,
                edgecolor="green",
                facecolor="green",
                alpha=1.0,
            )
            ax.add_patch(block_rect)

        ax.set_title(f"Cover2D Environment (Step {self.steps})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return ax
