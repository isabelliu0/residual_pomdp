"""Particle filter operations for tabletop belief tracking."""

from __future__ import annotations

import numpy as np

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
        unknown_objects=unknown_objects,
        visibility_grid=visibility_grid,
    )


def predict_belief(
    belief: Belief,
    action: np.ndarray,
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    noise_std: float = 0.01,
) -> Belief:
    """Predict belief forward after action.

    Applies action to joint positions (first 7 joints) and adds control
    noise. Objects remain static (no dynamics unless grasped).
    """
    new_particles = []
    for particle in belief.particles:
        current_joints = np.array(particle.joint_positions[:7], dtype=np.float32)

        new_joints = current_joints + action
        new_joints = np.clip(new_joints, joint_lower_limits, joint_upper_limits)

        noisy_joints = new_joints + np.random.normal(0, noise_std, size=7)
        noisy_joints = np.clip(noisy_joints, joint_lower_limits, joint_upper_limits)

        full_joints = tuple(noisy_joints) + particle.joint_positions[7:]

        new_particle = TabletopState(
            joint_positions=full_joints,
            gripper_open=particle.gripper_open,
            object_poses=particle.object_poses,
        )
        new_particles.append(new_particle)

    return Belief(
        particles=new_particles,
        weights=belief.weights.copy(),
        known_objects=belief.known_objects.copy(),
        unknown_objects=belief.unknown_objects.copy(),
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
    """Update belief from new camera observation."""
    detections = detect_objects_from_segmentation(
        camera_image.rgb,
        camera_image.depth,
        camera_image.segmentation,
        camera_pose,
        camera_intrinsics,
        object_ids,
        physics_client_id,
    )

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

    known_objects = set(detections.keys())
    all_objects_set = set(object_ids)
    unknown_objects = all_objects_set - known_objects

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
        known_objects=known_objects,
        unknown_objects=unknown_objects,
        visibility_grid=belief.visibility_grid,
    )


def get_mean_state(belief: Belief) -> TabletopState:
    """Extract mean state from belief (weighted average)."""
    mean_joints = np.average(
        [p.joint_positions for p in belief.particles], axis=0, weights=belief.weights
    )

    mean_gripper = np.average(
        [p.gripper_open for p in belief.particles], weights=belief.weights
    )

    all_obj_ids: set[int] = set()
    for p in belief.particles:
        all_obj_ids.update(p.object_poses.keys())

    mean_object_poses: dict[int, SE3Pose | None] = {}
    for obj_id in all_obj_ids:
        poses = [p.object_poses.get(obj_id) for p in belief.particles]
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


def average_poses(poses: list[tuple[float, ...]], weights: np.ndarray) -> SE3Pose:
    """Weighted average of SE(3) poses."""
    positions = np.array([(p[0], p[1], p[2]) for p in poses], dtype=np.float32)
    mean_pos = np.average(positions, axis=0, weights=weights)

    orientations = np.array([(p[3], p[4], p[5], p[6]) for p in poses], dtype=np.float32)
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
        [p.joint_positions for p in belief.particles], dtype=np.float32
    )
    joint_cov = np.cov(joint_particles.T, aweights=belief.weights)

    uncertainty = np.trace(joint_cov)
    confidence = np.exp(-uncertainty / threshold)

    return float(confidence)
