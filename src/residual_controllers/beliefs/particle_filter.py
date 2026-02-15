"""Particle filter operations for tabletop belief tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from residual_controllers.beliefs.occupancy_grid import LogOddsOccupancyGrid
from residual_controllers.beliefs.perception import detect_objects_from_segmentation
from residual_controllers.beliefs.structs import (
    Belief,
    BeliefConfig,
    CameraIntrinsics,
    SE3Pose,
    TabletopState,
)


@dataclass
class ObjectLikelihoodStats:
    """Per-object likelihood statistics."""

    obj_id: int
    mean_distance: float
    min_distance: float
    max_distance: float
    std_distance: float
    mean_likelihood: float
    min_likelihood: float
    max_likelihood: float
    particles_with_none_pose: int
    total_particles: int


@dataclass
class BeliefUpdateDiagnostics:
    """Diagnostics from a belief update step."""

    object_stats: dict[int, ObjectLikelihoodStats] = field(default_factory=dict)
    n_eff_before: float = 0.0
    n_eff_after: float = 0.0
    resampled: bool = False
    weight_entropy_before: float = 0.0
    weight_entropy_after: float = 0.0
    ego_pose_confidence: float = 1.0
    weight_std_before: float = 0.0
    weight_std_after: float = 0.0
    weight_max_before: float = 0.0
    weight_max_after: float = 0.0


def create_initial_belief(
    env,
    camera_image,
    num_particles: int = 100,
    config: BeliefConfig | None = None,
) -> Belief:
    """Initialize belief from first camera observation."""
    config = config or BeliefConfig()
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
    object_confidence = {obj_id: 0.0 for obj_id in all_objects}
    for obj_id in known_objects:
        object_confidence[obj_id] = 1.0

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
        bounds=[[0.2, 0.8], [-0.4, 0.4], [0.0, 0.5]],
        resolution=0.015,
        thresholds=config,
    )
    visibility_grid.update_from_depth(
        camera_pose,
        camera_image.depth,
        camera_image.segmentation,
        env.scene.object_ids,
        env.camera_intrinsics,
        stride=config.grid_stride,
        free_space_margin=config.free_space_margin,
        free_update=config.free_update,
        occ_update=config.occ_update,
        object_thickness=config.object_thickness,
        dense_stride=config.dense_stride,
        dense_window=config.dense_window,
        depth_noise_scale=config.depth_noise_scale,
    )

    return Belief(
        particles=particles,
        weights=weights,
        known_objects=known_objects,
        occluded_objects=set(),
        unknown_objects=unknown_objects,
        object_confidence=object_confidence,
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
    noise. If holding an object, updates its pose based on gripper
    motion.
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
        object_confidence=dict(belief.object_confidence),
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
    config: BeliefConfig | None = None,
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

    config = config or BeliefConfig()
    detected_objects = set(detections.keys())
    occluded_objects: set[int] = set()
    visibility_status: dict[int, str] = {}

    previously_known = belief.known_objects | belief.occluded_objects

    for obj_id in object_ids:
        if obj_id in detected_objects:
            visibility_status[obj_id] = "detected"
            continue

        if obj_id == belief.held_object_id:
            visibility_status[obj_id] = "held"
            continue

        last_known_pose = _get_last_known_pose(belief, obj_id)

        if last_known_pose is None:
            visibility_status[obj_id] = "never_seen"
        elif belief.visibility_grid is not None:
            visible_fraction = _fraction_visible_particles(
                belief, obj_id, belief.visibility_grid
            )
            if visible_fraction >= config.visible_fraction_threshold:
                visibility_status[obj_id] = "visible_missing"
            elif obj_id in previously_known:
                visibility_status[obj_id] = "occluded"
                occluded_objects.add(obj_id)
            else:
                visibility_status[obj_id] = "never_seen"
        elif obj_id in previously_known:
            visibility_status[obj_id] = "occluded"
            occluded_objects.add(obj_id)
        else:
            visibility_status[obj_id] = "never_seen"

    object_confidence = dict(belief.object_confidence)
    for obj_id in object_ids:
        conf = object_confidence.get(obj_id, 0.0)
        status = visibility_status.get(obj_id, "never_seen")
        if status in ["detected", "held"]:
            conf = 1.0
        elif status == "occluded":
            conf *= config.occluded_decay
        elif status == "visible_missing":
            conf *= config.visible_missing_decay
        else:
            conf *= config.never_seen_decay
        object_confidence[obj_id] = float(conf)

    known_objects = {
        obj_id
        for obj_id, conf in object_confidence.items()
        if conf >= config.confidence_known_threshold
    }
    unknown_objects = set(object_ids) - known_objects

    likelihoods = []
    updated_particles = []
    mean_distance_sums: dict[int, float] = {}
    mean_distance_weights: dict[int, float] = {}
    pos_std = config.pose_injection_pos_std
    rot_std = config.pose_injection_rot_std
    reset_distance = config.pose_injection_reset_distance
    min_alpha = config.pose_injection_min_alpha
    for particle_index, particle in enumerate(belief.particles):
        likelihood = 1.0
        new_object_poses = dict(particle.object_poses)

        for obj_id, (detected_pose, _) in detections.items():
            particle_pose = new_object_poses.get(obj_id)

            if particle_pose is None:
                likelihood *= config.missing_pose_likelihood
                new_object_poses[obj_id] = add_pose_noise(
                    detected_pose, pos_std=pos_std, rot_std=rot_std
                )
            else:
                dist = pose_distance(particle_pose, detected_pose)
                likelihood *= np.exp(
                    -0.5 * (dist / config.likelihood_distance_scale) ** 2
                )
                weight = float(belief.weights[particle_index])
                mean_distance_sums[obj_id] = mean_distance_sums.get(obj_id, 0.0) + (
                    weight * dist
                )
                mean_distance_weights[obj_id] = mean_distance_weights.get(
                    obj_id, 0.0
                ) + (weight)
                alpha = float(np.clip(dist / reset_distance, min_alpha, 1.0))
                blended_pose = blend_poses(particle_pose, detected_pose, alpha)
                new_object_poses[obj_id] = add_pose_noise(
                    blended_pose, pos_std=pos_std, rot_std=rot_std
                )

        for obj_id in occluded_objects:
            likelihood *= config.occluded_likelihood

        updated_particles.append(
            TabletopState(
                joint_positions=particle.joint_positions,
                gripper_open=particle.gripper_open,
                object_poses=new_object_poses,
            )
        )
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
        new_particles = [updated_particles[i] for i in indices]
        new_weights = np.ones(len(belief.particles), dtype=np.float64) / len(
            belief.particles
        )
    else:
        new_particles = updated_particles

    new_weights = new_weights.astype(np.float32)

    if belief.visibility_grid is not None:
        mean_distances = {
            obj_id: dist_sum / mean_distance_weights[obj_id]
            for obj_id, dist_sum in mean_distance_sums.items()
            if mean_distance_weights.get(obj_id, 0.0) > 0.0
        }
        observation_confidence = compute_observation_confidence(
            belief,
            detections,
            mean_distances=mean_distances,
            distance_scale=config.likelihood_distance_scale,
        )
        ego_confidence = compute_ego_pose_confidence(
            belief,
            threshold=config.ego_pose_confidence_threshold,
            observation_confidence=observation_confidence,
            boost_scale=config.ego_pose_boost_scale,
        )
        belief.visibility_grid.update_from_depth(
            camera_pose,
            camera_image.depth,
            camera_image.segmentation,
            object_ids,
            camera_intrinsics,
            ego_confidence,
            stride=config.grid_stride,
            free_space_margin=config.free_space_margin,
            free_update=config.free_update,
            occ_update=config.occ_update,
            object_thickness=config.object_thickness,
            dense_stride=config.dense_stride,
            dense_window=config.dense_window,
            depth_noise_scale=config.depth_noise_scale,
        )

    return Belief(
        particles=new_particles,
        weights=new_weights,
        known_objects=known_objects,
        occluded_objects=occluded_objects,
        unknown_objects=unknown_objects,
        object_confidence=object_confidence,
        held_object_id=belief.held_object_id,
        visibility_grid=belief.visibility_grid,
    )


def compute_belief_diagnostics(
    belief: Belief,
    camera_image,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    camera_intrinsics: CameraIntrinsics,
    object_ids: list[int],
    physics_client_id: int,
    config: BeliefConfig | None = None,
) -> BeliefUpdateDiagnostics:
    """Compute diagnostics for belief update without modifying belief."""
    detections = detect_objects_from_segmentation(
        camera_image.rgb,
        camera_image.depth,
        camera_image.segmentation,
        camera_pose,
        camera_intrinsics,
        object_ids,
        physics_client_id,
    )

    diagnostics = BeliefUpdateDiagnostics()
    config = config or BeliefConfig()

    weights = belief.weights.astype(np.float64)
    weights = weights / weights.sum()
    diagnostics.weight_entropy_before = float(
        -np.sum(weights * np.log(weights + 1e-10))
    )
    diagnostics.n_eff_before = float(1.0 / (weights**2).sum())
    diagnostics.weight_std_before = float(np.std(weights))
    diagnostics.weight_max_before = float(np.max(weights))

    # Per-object statistics
    for obj_id, (detected_pose, _) in detections.items():
        distances = []
        likelihoods = []
        none_count = 0

        for particle in belief.particles:
            particle_pose = particle.object_poses.get(obj_id)
            if particle_pose is None:
                none_count += 1
                likelihoods.append(0.5)
            else:
                dist = pose_distance(particle_pose, detected_pose)
                distances.append(dist)
                likelihood = np.exp(
                    -0.5 * (dist / config.likelihood_distance_scale) ** 2
                )
                likelihoods.append(likelihood)

        stats = ObjectLikelihoodStats(
            obj_id=obj_id,
            mean_distance=float(np.mean(distances)) if distances else 0.0,
            min_distance=float(np.min(distances)) if distances else 0.0,
            max_distance=float(np.max(distances)) if distances else 0.0,
            std_distance=float(np.std(distances)) if distances else 0.0,
            mean_likelihood=float(np.mean(likelihoods)),
            min_likelihood=float(np.min(likelihoods)),
            max_likelihood=float(np.max(likelihoods)),
            particles_with_none_pose=none_count,
            total_particles=len(belief.particles),
        )
        diagnostics.object_stats[obj_id] = stats

    # Compute what n_eff would be after update
    likelihoods_all = []
    for particle in belief.particles:
        likelihood = 1.0
        for obj_id, (detected_pose, _) in detections.items():
            particle_pose = particle.object_poses.get(obj_id)
            if particle_pose is None:
                likelihood *= config.missing_pose_likelihood
            else:
                dist = pose_distance(particle_pose, detected_pose)
                likelihood *= np.exp(
                    -0.5 * (dist / config.likelihood_distance_scale) ** 2
                )
        likelihoods_all.append(likelihood)

    likelihoods_arr = np.array(likelihoods_all, dtype=np.float64)
    new_weights = weights * likelihoods_arr
    weight_sum = new_weights.sum()
    if weight_sum > 1e-100:
        new_weights = new_weights / weight_sum
        diagnostics.n_eff_after = float(1.0 / (new_weights**2).sum())
        diagnostics.weight_entropy_after = float(
            -np.sum(new_weights * np.log(new_weights + 1e-10))
        )
        diagnostics.weight_std_after = float(np.std(new_weights))
        diagnostics.weight_max_after = float(np.max(new_weights))
        diagnostics.resampled = diagnostics.n_eff_after < len(belief.particles) / 2

    mean_distances = {
        obj_id: stats.mean_distance
        for obj_id, stats in diagnostics.object_stats.items()
        if stats.particles_with_none_pose < stats.total_particles
    }
    observation_confidence = compute_observation_confidence(
        belief,
        detections,
        mean_distances=mean_distances,
        distance_scale=config.likelihood_distance_scale,
    )
    diagnostics.ego_pose_confidence = compute_ego_pose_confidence(
        belief,
        threshold=config.ego_pose_confidence_threshold,
        observation_confidence=observation_confidence,
        boost_scale=config.ego_pose_boost_scale,
    )

    return diagnostics


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
    return visibility_grid.is_free((position[0], position[1], position[2]))


def _fraction_visible_particles(
    belief: Belief, obj_id: int, visibility_grid: LogOddsOccupancyGrid
) -> float:
    """Fraction of particles whose object pose lies in visible grid space."""
    visible = 0
    total = 0
    for particle in belief.particles:
        pose = particle.object_poses.get(obj_id)
        if pose is None:
            continue
        total += 1
        if _is_position_visible(visibility_grid, pose[:3]):
            visible += 1
    if total == 0:
        return 0.0
    return float(visible) / float(total)


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


def blend_poses(
    pose_a: tuple[float, ...],
    pose_b: tuple[float, ...],
    alpha: float,
) -> SE3Pose:
    """Blend between two poses with linear interpolation and normalized
    quats."""
    ax, ay, az, aqx, aqy, aqz, aqw = pose_a
    bx, by, bz, bqx, bqy, bqz, bqw = pose_b

    x = (1.0 - alpha) * ax + alpha * bx
    y = (1.0 - alpha) * ay + alpha * by
    z = (1.0 - alpha) * az + alpha * bz

    qx = (1.0 - alpha) * aqx + alpha * bqx
    qy = (1.0 - alpha) * aqy + alpha * bqy
    qz = (1.0 - alpha) * aqz + alpha * bqz
    qw = (1.0 - alpha) * aqw + alpha * bqw
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-8:
        return (x, y, z, aqx, aqy, aqz, aqw)

    return (x, y, z, qx / norm, qy / norm, qz / norm, qw / norm)


def compute_observation_confidence(
    belief: Belief,
    detections: dict[int, tuple[tuple[float, ...], float]],
    mean_distances: dict[int, float] | None = None,
    distance_scale: float = 0.15,
) -> float:
    """Score observation consistency based on particle pose agreement."""
    if not detections:
        return 0.0

    if mean_distances is not None:
        if not mean_distances:
            return 0.0
        scores = [
            float(np.exp(-0.5 * (mean_dist / distance_scale) ** 2))
            for mean_dist in mean_distances.values()
        ]
        return float(np.mean(scores))

    scores = []
    for obj_id, (detected_pose, _) in detections.items():
        distances = []
        weights = []
        for i, particle in enumerate(belief.particles):
            particle_pose = particle.object_poses.get(obj_id)
            if particle_pose is None:
                continue
            distances.append(pose_distance(particle_pose, detected_pose))
            weights.append(float(belief.weights[i]))
        if not distances:
            continue
        weights_arr = np.array(weights, dtype=np.float64)
        weights_arr /= weights_arr.sum()
        mean_dist = float(np.sum(np.array(distances) * weights_arr))
        scores.append(float(np.exp(-0.5 * (mean_dist / distance_scale) ** 2)))

    if not scores:
        return 0.0

    return float(np.mean(scores))


def compute_ego_pose_confidence(
    belief: Belief,
    threshold: float = 0.05,
    observation_confidence: float | None = None,
    boost_scale: float = 0.75,
) -> float:
    """Compute confidence in current camera pose from joint uncertainty."""
    joint_particles = np.array(
        [particle.joint_positions for particle in belief.particles], dtype=np.float32
    )
    joint_cov = np.cov(joint_particles.T, aweights=belief.weights)

    uncertainty = np.trace(joint_cov)
    confidence = np.exp(-uncertainty / threshold)
    if observation_confidence is not None:
        confidence *= 1.0 + boost_scale * observation_confidence
        confidence = min(1.0, confidence)

    return float(confidence)
