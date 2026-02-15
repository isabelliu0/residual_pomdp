"""Particle filter operations for tabletop belief tracking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pybullet as p

from residual_controllers.beliefs.occupancy_grid import LogOddsOccupancyGrid
from residual_controllers.beliefs.perception import detect_objects_from_segmentation
from residual_controllers.beliefs.structs import (
    Belief,
    CameraIntrinsics,
    SE3Pose,
    TabletopState,
)


def create_initial_belief(
    env,
    camera_image,
    num_particles: int = 100,
) -> Belief:
    """Initialize belief from first camera observation."""
    camera_pose = env.get_camera_pose_se3()
    detections = detect_objects_from_segmentation(
        camera_image.rgb,
        camera_image.depth,
        camera_image.segmentation,
        camera_pose,
        env.camera_intrinsics,
        env.scene.object_ids,
        env.physics_client_id,
    )

    known_objects = set(detections.keys())
    all_objects = set(env.scene.object_ids)
    unknown_objects = all_objects - known_objects

    particles = []
    for _ in range(num_particles):
        joint_positions = tuple(env.robot.get_joint_positions())
        gripper_open = 0.04

        object_poses: dict[int, SE3Pose | None] = {}

        for obj_id in known_objects:
            detected_pose, _ = detections[obj_id]
            noisy_pose = add_pose_noise(detected_pose, pos_std=0.01, rot_std=0.09)
            object_poses[obj_id] = noisy_pose

        for obj_id in unknown_objects:
            object_poses[obj_id] = None

        state = TabletopState(
            joint_positions=joint_positions,
            gripper_open=gripper_open,
            object_poses=object_poses,
        )
        particles.append(state)

    weights = np.ones(num_particles, dtype=np.float32) / num_particles

    visibility_grid = LogOddsOccupancyGrid(
        bounds=[[0.2, 0.8], [-0.4, 0.4], [0.0, 0.5]], resolution=0.015
    )
    visibility_grid.update_from_depth(
        camera_pose, camera_image.depth, env.camera_intrinsics, stride=20
    )

    return Belief(
        particles=particles,
        weights=weights,
        known_objects=known_objects,
        occluded_objects=set(),
        unknown_objects=unknown_objects,
        held_object_id=None,
        visibility_grid=visibility_grid,
    )


def predict_belief(
    belief: Belief,
    action: np.ndarray,
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    ee_pose_before: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    ee_pose_after: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    noise_std: float = 0.01,
) -> Belief:
    """Predict belief forward after action.

    Applies action to joint positions (first 7 joints) and adds control
    noise. If holding an object, updates its pose based on gripper motion.

    Args:
        belief: Current belief state
        action: Joint position deltas (7D)
        joint_lower_limits: Lower joint limits
        joint_upper_limits: Upper joint limits
        ee_pose_before: End-effector pose before action (for held object update)
        ee_pose_after: End-effector pose after action (for held object update)
        noise_std: Noise standard deviation for joint positions
    """
    new_particles = []
    held_id = belief.held_object_id

    for particle in belief.particles:
        current_joints = np.array(particle.joint_positions[:7], dtype=np.float32)

        new_joints = current_joints + action
        new_joints = np.clip(new_joints, joint_lower_limits, joint_upper_limits)

        noisy_joints = new_joints + np.random.normal(0, noise_std, size=7)
        noisy_joints = np.clip(noisy_joints, joint_lower_limits, joint_upper_limits)

        full_joints = tuple(noisy_joints) + particle.joint_positions[7:]

        new_object_poses = particle.object_poses.copy()

        if (
            held_id is not None
            and ee_pose_before is not None
            and ee_pose_after is not None
        ):
            held_pose = particle.object_poses.get(held_id)
            if held_pose is not None:
                new_held_pose = transform_held_object_pose(
                    held_pose, ee_pose_before, ee_pose_after
                )
                new_object_poses[held_id] = new_held_pose

        new_particle = TabletopState(
            joint_positions=full_joints,
            gripper_open=particle.gripper_open,
            object_poses=new_object_poses,
        )
        new_particles.append(new_particle)

    return Belief(
        particles=new_particles,
        weights=belief.weights.copy(),
        known_objects=belief.known_objects.copy(),
        occluded_objects=belief.occluded_objects.copy(),
        unknown_objects=belief.unknown_objects.copy(),
        held_object_id=belief.held_object_id,
        visibility_grid=belief.visibility_grid,
    )


def update_belief(
    belief: Belief,
    camera_image,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    camera_intrinsics: CameraIntrinsics,
    object_ids: list[int],
    physics_client_id: int,
) -> Belief:
    """Update belief from new camera observation.

    Uses visibility grid to properly handle occlusion:
    - detected_objects: Currently visible and detected
    - occluded_objects: Previously known, now in occluded region (maintain pose)
    - unknown_objects: Never detected or expected visible but not found
    """
    detections = detect_objects_from_segmentation(
        camera_image.rgb,
        camera_image.depth,
        camera_image.segmentation,
        camera_pose,
        camera_intrinsics,
        object_ids,
        physics_client_id,
    )

    detected_objects = set(detections.keys())
    occluded_objects: set[int] = set()
    unknown_objects: set[int] = set()

    previously_known = belief.known_objects | belief.occluded_objects

    for obj_id in object_ids:
        if obj_id in detected_objects:
            continue

        if obj_id == belief.held_object_id:
            continue

        last_known_pose = _get_last_known_pose(belief, obj_id)

        if last_known_pose is None:
            unknown_objects.add(obj_id)
        elif belief.visibility_grid is not None and _is_position_visible(
            belief.visibility_grid, last_known_pose[:3]
        ):
            unknown_objects.add(obj_id)
        elif obj_id in previously_known:
            occluded_objects.add(obj_id)
        else:
            unknown_objects.add(obj_id)

    likelihoods = []
    for particle in belief.particles:
        likelihood = 1.0

        for obj_id, (detected_pose, _) in detections.items():
            particle_pose = particle.object_poses.get(obj_id)

            if particle_pose is None:
                likelihood *= 0.5
            else:
                dist = pose_distance(particle_pose, detected_pose)
                likelihood *= np.exp(-0.5 * (dist / 0.15) ** 2)

        for obj_id in occluded_objects:
            likelihood *= 0.95

        likelihoods.append(likelihood)

    likelihoods_arr = np.array(likelihoods, dtype=np.float64)
    new_weights = belief.weights.astype(np.float64) * likelihoods_arr

    weight_sum = new_weights.sum()
    if weight_sum < 1e-100:
        new_weights = np.ones(len(belief.particles), dtype=np.float64) / len(
            belief.particles
        )
    else:
        new_weights = new_weights / weight_sum

    n_eff = 1.0 / (new_weights**2).sum()
    if n_eff < len(belief.particles) / 2:
        new_weights_normalized = new_weights / new_weights.sum()
        indices = np.random.choice(
            len(belief.particles), size=len(belief.particles), p=new_weights_normalized
        )
        new_particles = [belief.particles[i] for i in indices]
        new_weights = np.ones(len(belief.particles), dtype=np.float64) / len(
            belief.particles
        )
    else:
        new_particles = belief.particles

    new_weights = new_weights.astype(np.float32)

    if belief.visibility_grid is not None:
        ego_confidence = compute_ego_pose_confidence(belief)
        belief.visibility_grid.update_from_depth(
            camera_pose,
            camera_image.depth,
            camera_intrinsics,
            ego_confidence,
            stride=20,
        )

    return Belief(
        particles=new_particles,
        weights=new_weights,
        known_objects=detected_objects,
        occluded_objects=occluded_objects,
        unknown_objects=unknown_objects,
        held_object_id=belief.held_object_id,
        visibility_grid=belief.visibility_grid,
    )


def _get_last_known_pose(belief: Belief, obj_id: int) -> tuple[float, ...] | None:
    """Get last known pose for an object from particles (weighted mean)."""
    poses = []
    weights = []
    for i, particle in enumerate(belief.particles):
        pose = particle.object_poses.get(obj_id)
        if pose is not None:
            poses.append(pose)
            weights.append(belief.weights[i])

    if not poses:
        return None

    weights_arr = np.array(weights)
    weights_arr /= weights_arr.sum()

    return average_poses(poses, weights_arr)


def _is_position_visible(
    visibility_grid: LogOddsOccupancyGrid, position: tuple[float, ...]
) -> bool:
    """Check if position is in a visible (free) region of the grid."""
    return not visibility_grid.is_occupied((position[0], position[1], position[2]))


def get_mean_state(belief: Belief) -> TabletopState:
    """Extract mean state from belief (weighted average)."""
    mean_joints = np.average(
        [particle.joint_positions for particle in belief.particles],
        axis=0,
        weights=belief.weights,
    )

    mean_gripper = np.average(
        [particle.gripper_open for particle in belief.particles], weights=belief.weights
    )

    all_obj_ids: set[int] = set()
    for particle in belief.particles:
        all_obj_ids.update(particle.object_poses.keys())

    mean_object_poses: dict[int, SE3Pose | None] = {}
    for obj_id in all_obj_ids:
        poses = [particle.object_poses.get(obj_id) for particle in belief.particles]
        if all(pose is not None for pose in poses):
            mean_pose = average_poses(poses, belief.weights)  # type: ignore
            mean_object_poses[obj_id] = mean_pose
        else:
            mean_object_poses[obj_id] = None

    return TabletopState(
        joint_positions=tuple(mean_joints),
        gripper_open=float(mean_gripper),
        object_poses=mean_object_poses,
    )


def add_pose_noise(pose: tuple[float, ...], pos_std: float, rot_std: float) -> SE3Pose:
    """Add Gaussian noise to SE(3) pose."""
    x, y, z, qx, qy, qz, qw = pose
    x += np.random.normal(0, pos_std)
    y += np.random.normal(0, pos_std)
    z += np.random.normal(0, pos_std)

    qx += np.random.normal(0, rot_std)
    qy += np.random.normal(0, rot_std)
    qz += np.random.normal(0, rot_std)
    qw += np.random.normal(0, rot_std)
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)

    return (x, y, z, qx / norm, qy / norm, qz / norm, qw / norm)


def pose_distance(pose1: tuple[float, ...], pose2: tuple[float, ...]) -> float:
    """Euclidean distance between two SE(3) poses (position only)."""
    x1, y1, z1 = pose1[0], pose1[1], pose1[2]
    x2, y2, z2 = pose2[0], pose2[1], pose2[2]
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))


def transform_held_object_pose(
    object_pose: tuple[float, ...],
    ee_pose_before: tuple[tuple[float, ...], tuple[float, ...]],
    ee_pose_after: tuple[tuple[float, ...], tuple[float, ...]],
) -> SE3Pose:
    """Transform held object pose based on gripper motion.

    Computes how the object moves given the change in end-effector pose,
    assuming the object is rigidly attached to the gripper.
    """
    obj_pos = np.array(object_pose[:3])
    obj_quat = object_pose[3:7]

    ee_pos_before = np.array(ee_pose_before[0])
    ee_quat_before = ee_pose_before[1]
    ee_pos_after = np.array(ee_pose_after[0])
    ee_quat_after = ee_pose_after[1]

    ee_rot_before = np.array(p.getMatrixFromQuaternion(ee_quat_before)).reshape(3, 3)
    ee_rot_after = np.array(p.getMatrixFromQuaternion(ee_quat_after)).reshape(3, 3)

    obj_in_ee = ee_rot_before.T @ (obj_pos - ee_pos_before)

    new_obj_pos = ee_rot_after @ obj_in_ee + ee_pos_after

    _, ee_quat_before_inv = p.invertTransform([0, 0, 0], ee_quat_before)
    _, obj_in_ee_quat = p.multiplyTransforms(
        [0, 0, 0], ee_quat_before_inv, [0, 0, 0], obj_quat
    )
    _, new_obj_quat = p.multiplyTransforms(
        [0, 0, 0], ee_quat_after, [0, 0, 0], obj_in_ee_quat
    )

    return (
        float(new_obj_pos[0]),
        float(new_obj_pos[1]),
        float(new_obj_pos[2]),
        float(new_obj_quat[0]),
        float(new_obj_quat[1]),
        float(new_obj_quat[2]),
        float(new_obj_quat[3]),
    )


def average_poses(poses: Sequence[tuple[float, ...]], weights: np.ndarray) -> SE3Pose:
    """Weighted average of SE(3) poses."""
    positions = np.array(
        [(pose[0], pose[1], pose[2]) for pose in poses], dtype=np.float32
    )
    mean_pos = np.average(positions, axis=0, weights=weights)

    orientations = np.array(
        [(pose[3], pose[4], pose[5], pose[6]) for pose in poses], dtype=np.float32
    )
    mean_quat = np.average(orientations, axis=0, weights=weights)
    mean_quat /= np.linalg.norm(mean_quat)

    return (
        float(mean_pos[0]),
        float(mean_pos[1]),
        float(mean_pos[2]),
        float(mean_quat[0]),
        float(mean_quat[1]),
        float(mean_quat[2]),
        float(mean_quat[3]),
    )


def compute_ego_pose_confidence(belief: Belief, threshold: float = 0.01) -> float:
    """Compute confidence in current camera pose from joint uncertainty."""
    joint_particles = np.array(
        [particle.joint_positions for particle in belief.particles], dtype=np.float32
    )
    joint_cov = np.cov(joint_particles.T, aweights=belief.weights)

    uncertainty = np.trace(joint_cov)
    confidence = np.exp(-uncertainty / threshold)

    return float(confidence)
