"""Particle filter operations for tabletop belief tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pybullet as p

from residual_controllers.beliefs.occupancy_grid import LogOddsOccupancyGrid
from residual_controllers.beliefs.perception import (
    detect_objects_from_pointcloud_pca,
    detect_objects_from_segmentation,
)
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

    label: str
    mean_distance: float
    min_distance: float
    max_distance: float
    std_distance: float
    mean_likelihood: float
    min_likelihood: float
    max_likelihood: float
    total_particles: int


@dataclass
class BeliefUpdateDiagnostics:
    """Diagnostics from a belief update step."""

    object_stats: dict[str, ObjectLikelihoodStats] = field(default_factory=dict)
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


def _precompute_surface_xy(
    visibility_grid: LogOddsOccupancyGrid,
    table_id: int,
    physics_client_id: int,
    excluded_aabbs: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Precompute unseen surface (x, y) candidates from the occupancy grid.

    Call this once before a particle resampling loop and pass the result
    to _sample_on_table_pose to avoid recomputing the sigmoid over all
    voxels for every particle.
    """
    table_aabb = p.getAABB(table_id, physicsClientId=physics_client_id)
    table_top_z = float(table_aabb[1][2])
    unseen = visibility_grid.get_unobserved_voxels()
    surface_xy = []
    for vx, vy, vz in unseen:
        if not abs(vz - table_top_z) < 0.05:
            continue
        if not table_aabb[0][0] <= vx <= table_aabb[1][0]:
            continue
        if not table_aabb[0][1] <= vy <= table_aabb[1][1]:
            continue
        if excluded_aabbs and any(
            xmin <= vx <= xmax and ymin <= vy <= ymax
            for xmin, ymin, xmax, ymax in excluded_aabbs
        ):
            continue
        surface_xy.append((vx, vy))
    # print(f"[surface_xy] {len(surface_xy)} unseen surface voxels (of {len(unseen)} total unseen)")    # pylint: disable=line-too-long
    # if surface_xy:
    #     xs = np.array([v[0] for v in surface_xy])
    #     ys = np.array([v[1] for v in surface_xy])
    #     x_bins = np.linspace(xs.min(), xs.max(), 6)
    #     y_bins = np.linspace(ys.min(), ys.max(), 6)
    #     hist, _, _ = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    #     print(f"[surface_xy] 2D count distribution (x=rows, y=cols):")
    #     print(f"[surface_xy]   x=[{xs.min():.3f},{xs.max():.3f}]  y=[{ys.min():.3f},{ys.max():.3f}]") # pylint: disable=line-too-long
    #     for row in hist.astype(int):
    #         print(f"[surface_xy]   {row.tolist()}")
    return surface_xy


def _sample_on_table_pose(
    obj_id: int,
    table_id: int,
    physics_client_id: int,
    precomputed_surface_xy: list[tuple[float, float]] | None = None,
) -> SE3Pose:
    """Sample a random pose for obj placed flat on the table surface.

    Pass precomputed_surface_xy to bias sampling toward unseen voxels.
    Falls back to uniform sampling over the table.
    """
    table_aabb = p.getAABB(table_id, physicsClientId=physics_client_id)
    table_top_z = float(table_aabb[1][2])
    obj_aabb = p.getAABB(obj_id, physicsClientId=physics_client_id)
    obj_half_z = float((obj_aabb[1][2] - obj_aabb[0][2]) / 2.0)
    z = table_top_z + obj_half_z

    if precomputed_surface_xy:
        vx, vy = precomputed_surface_xy[np.random.randint(len(precomputed_surface_xy))]
        return (float(vx), float(vy), z, 0.0, 0.0, 0.0, 1.0)

    x = float(np.random.uniform(table_aabb[0][0], table_aabb[1][0]))
    y = float(np.random.uniform(table_aabb[0][1], table_aabb[1][1]))
    return (x, y, z, 0.0, 0.0, 0.0, 1.0)


def _best_yaw_equivalent(
    particle_pose: tuple[float, ...],
    detected_pose: tuple[float, ...],
    n_fold: int,
) -> SE3Pose:
    """Return the n-fold equivalent of detected_pose whose yaw is closest to
    particle_pose."""
    if n_fold <= 1:
        return cast(SE3Pose, detected_pose)
    x, y, z = detected_pose[0], detected_pose[1], detected_pose[2]
    yaw_det = 2.0 * float(np.arctan2(detected_pose[5], detected_pose[6]))
    yaw_p = 2.0 * float(np.arctan2(particle_pose[5], particle_pose[6]))
    period = 2.0 * np.pi / n_fold
    best_yaw = yaw_det
    best_diff = abs(float((yaw_det - yaw_p + np.pi) % (2 * np.pi) - np.pi))
    for k in range(1, n_fold):
        yaw_k = yaw_det + k * period
        diff = abs(float((yaw_k - yaw_p + np.pi) % (2 * np.pi) - np.pi))
        if diff < best_diff:
            best_diff = diff
            best_yaw = yaw_k
    return (x, y, z, 0.0, 0.0, float(np.sin(best_yaw / 2)), float(np.cos(best_yaw / 2)))


def create_initial_belief(
    env,
    camera_image,
    num_particles: int = 100,
    config: BeliefConfig | None = None,
) -> Belief:
    """Initialize belief from first camera observation."""
    config = config or BeliefConfig()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env.scene.label_to_id
    detections = detect_objects_from_pointcloud_pca(
        camera_image.segmentation,
        camera_image.depth,
        label_to_id,
        env.physics_client_id,
        env.camera_intrinsics,
        camera_pose,
        pixel_noise_std=config.pixel_noise_std,
    )

    known_objects: set[str] = set(detections.keys())
    all_objects: set[str] = set(label_to_id.keys())
    unknown_objects: set[str] = all_objects - known_objects
    object_confidence: dict[str, float] = {label: 0.0 for label in all_objects}
    for label in known_objects:
        object_confidence[label] = 1.0

    visibility_grid = LogOddsOccupancyGrid(
        bounds=config.grid_bounds,
        resolution=config.grid_resolution,
        thresholds=config,
    )
    visibility_grid.update_from_depth(
        camera_pose,
        camera_image.depth,
        camera_image.segmentation,
        list(label_to_id.values()),
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

    excluded_aabbs: list[tuple[float, float, float, float]] = []
    if (
        hasattr(env, "scene")
        and env.scene is not None
        and hasattr(env.scene, "target_area_id")
    ):
        aabb = p.getAABB(
            env.scene.target_area_id, physicsClientId=env.physics_client_id
        )
        excluded_aabbs.append(
            (float(aabb[0][0]), float(aabb[0][1]), float(aabb[1][0]), float(aabb[1][1]))
        )

    for label in known_objects:
        obj_id = label_to_id[label]
        aabb = p.getAABB(obj_id, physicsClientId=env.physics_client_id)
        excluded_aabbs.append(
            (float(aabb[0][0]), float(aabb[0][1]), float(aabb[1][0]), float(aabb[1][1]))
        )

    surface_xy: list[tuple[float, float]] | None = None
    if unknown_objects and hasattr(env.scene, "table_id"):
        surface_xy = _precompute_surface_xy(
            visibility_grid, env.scene.table_id, env.physics_client_id, excluded_aabbs
        )

    particles = []
    for i in range(num_particles):
        object_poses: dict[str, SE3Pose] = {}

        for label in known_objects:
            detected_pose, _ = detections[label]
            object_poses[label] = (
                cast(SE3Pose, detected_pose)
                if i == 0
                else add_pose_noise(detected_pose, pos_std=0.01, rot_std=0.0)
            )

        for label in unknown_objects:
            obj_id = label_to_id[label]
            object_poses[label] = _sample_on_table_pose(
                obj_id,
                env.scene.table_id,
                env.physics_client_id,
                precomputed_surface_xy=surface_xy,
            )

        particles.append(
            TabletopState(
                joint_positions=tuple(env.robot.get_joint_positions()),
                gripper_open=0.04,
                object_poses=object_poses,
            )
        )

    weights = np.ones(num_particles, dtype=np.float32) / num_particles

    return Belief(
        particles=particles,
        weights=weights,
        known_objects=known_objects,
        occluded_objects=set(),
        unknown_objects=unknown_objects,
        object_confidence=object_confidence,
        held_object_label=None,
        visibility_grid=visibility_grid,
    )


def predict_belief(
    belief: Belief,
    action: np.ndarray,
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    noise_std: float = 0.01,
    held_object_gt_pose: SE3Pose | None = None,
) -> Belief:
    """Predict belief forward after action.

    Applies action to joint positions (first 7 joints) and adds control
    noise. Static object poses are unchanged.
    """
    new_particles = []
    held_label = belief.held_object_label

    for particle in belief.particles:
        current_joints = np.array(particle.joint_positions[:7], dtype=np.float32)

        new_joints = current_joints + action
        new_joints = np.clip(new_joints, joint_lower_limits, joint_upper_limits)

        noisy_joints = new_joints + np.random.normal(0, noise_std, size=7)
        noisy_joints = np.clip(noisy_joints, joint_lower_limits, joint_upper_limits)

        full_joints = tuple(noisy_joints) + particle.joint_positions[7:]

        new_object_poses = particle.object_poses.copy()
        if held_label is not None and held_object_gt_pose is not None:
            if held_label in new_object_poses:
                new_object_poses[held_label] = held_object_gt_pose

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
        held_object_label=belief.held_object_label,
        visibility_grid=belief.visibility_grid,
    )


def update_belief(
    belief: Belief,
    camera_image,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    camera_intrinsics: CameraIntrinsics,
    label_to_id: dict[str, int],
    physics_client_id: int,
    config: BeliefConfig | None = None,
    table_id: int | None = None,
    excluded_aabbs: list[tuple[float, float, float, float]] | None = None,
) -> Belief:
    """Update belief from new camera observation."""
    config = config or BeliefConfig()
    detections = detect_objects_from_pointcloud_pca(
        camera_image.segmentation,
        camera_image.depth,
        label_to_id,
        physics_client_id,
        camera_intrinsics,
        camera_pose,
        pixel_noise_std=config.pixel_noise_std,
    )
    detected_objects = set(detections.keys())
    occluded_objects: set[str] = set()
    visibility_status: dict[str, str] = {}

    previously_known = belief.known_objects | belief.occluded_objects

    for label in label_to_id:
        if label in detected_objects:
            visibility_status[label] = "detected"
            continue

        if belief.visibility_grid is not None:
            visible_fraction = _fraction_visible_particles(
                belief, label, belief.visibility_grid
            )
            if visible_fraction >= config.visible_fraction_threshold:
                visibility_status[label] = "visible_missing"
            elif label in previously_known:
                visibility_status[label] = "occluded"
                occluded_objects.add(label)
            else:
                visibility_status[label] = "never_seen"
        elif label in previously_known:
            visibility_status[label] = "occluded"
            occluded_objects.add(label)
        else:
            visibility_status[label] = "never_seen"

    object_confidence = dict(belief.object_confidence)
    for label in label_to_id:
        conf = object_confidence.get(label, 0.0)
        status = visibility_status.get(label, "never_seen")
        if status in ["detected", "held"]:
            conf = 1.0
        elif status == "occluded":
            conf *= config.occluded_decay
        elif status == "visible_missing":
            conf *= config.visible_missing_decay
        else:
            conf *= config.never_seen_decay
        object_confidence[label] = float(conf)

    known_objects = {
        label
        for label, conf in object_confidence.items()
        if conf >= config.confidence_known_threshold
    }
    unknown_objects = set(label_to_id.keys()) - known_objects

    likelihoods = []
    updated_particles = []
    mean_distance_sums: dict[str, float] = {}
    mean_distance_weights: dict[str, float] = {}
    pos_std = config.pose_injection_pos_std
    rot_std = config.pose_injection_rot_std
    reset_distance = config.pose_injection_reset_distance
    min_alpha = config.pose_injection_min_alpha
    for particle_index, particle in enumerate(belief.particles):
        likelihood = 1.0
        new_object_poses = dict(particle.object_poses)

        for label, (detected_pose, _) in detections.items():
            if label in belief.unknown_objects:
                new_object_poses[label] = (
                    cast(SE3Pose, detected_pose)
                    if particle_index == 0
                    else add_pose_noise(detected_pose, pos_std=pos_std, rot_std=rot_std)
                )
            else:
                particle_pose = new_object_poses[label]
                dist = pose_distance(particle_pose, detected_pose)
                likelihood *= np.exp(
                    -0.5 * (dist / config.likelihood_distance_scale) ** 2
                )
                weight = float(belief.weights[particle_index])
                mean_distance_sums[label] = mean_distance_sums.get(label, 0.0) + (
                    weight * dist
                )
                mean_distance_weights[label] = mean_distance_weights.get(label, 0.0) + (
                    weight
                )
                alpha = float(np.clip(dist / reset_distance, min_alpha, 1.0))
                equiv_pose = _best_yaw_equivalent(
                    particle_pose, detected_pose, config.get_n_fold(label)
                )
                new_object_poses[label] = blend_poses(particle_pose, equiv_pose, alpha)

        for label in occluded_objects:
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
            label: dist_sum / mean_distance_weights[label]
            for label, dist_sum in mean_distance_sums.items()
            if mean_distance_weights.get(label, 0.0) > 0.0
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
            list(label_to_id.values()),
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

        if table_id is not None and unknown_objects:
            never_seen = unknown_objects - occluded_objects
            if never_seen:
                detected_excluded = list(excluded_aabbs or [])
                for label in detections:
                    aabb = p.getAABB(
                        label_to_id[label], physicsClientId=physics_client_id
                    )
                    detected_excluded.append(
                        (
                            float(aabb[0][0]),
                            float(aabb[0][1]),
                            float(aabb[1][0]),
                            float(aabb[1][1]),
                        )
                    )
                surface_xy = _precompute_surface_xy(
                    belief.visibility_grid,
                    table_id,
                    physics_client_id,
                    detected_excluded,
                )
                resampled = []
                for particle in new_particles:
                    new_poses = dict(particle.object_poses)
                    for label in never_seen:
                        if label not in detections:
                            new_poses[label] = _sample_on_table_pose(
                                label_to_id[label],
                                table_id,
                                physics_client_id,
                                precomputed_surface_xy=surface_xy,
                            )
                    resampled.append(
                        TabletopState(
                            joint_positions=particle.joint_positions,
                            gripper_open=particle.gripper_open,
                            object_poses=new_poses,
                        )
                    )
                new_particles = resampled

    return Belief(
        particles=new_particles,
        weights=new_weights,
        known_objects=known_objects,
        occluded_objects=occluded_objects,
        unknown_objects=unknown_objects,
        object_confidence=object_confidence,
        held_object_label=belief.held_object_label,
        visibility_grid=belief.visibility_grid,
    )


def update_belief_from_contact(
    belief: Belief,
    robot_body_id: int,
    label_to_id: dict[str, int],
    held_object_id: int | None,
    physics_client_id: int,
    config: BeliefConfig | None = None,
) -> Belief:
    """Blend particle poses toward GT for objects the gripper is touching but
    not holding."""
    config = config or BeliefConfig()

    contact_gt: dict[str, SE3Pose] = {}
    for label, obj_id in label_to_id.items():
        if obj_id == held_object_id:
            continue
        contacts = p.getContactPoints(
            bodyA=robot_body_id, bodyB=obj_id, physicsClientId=physics_client_id
        )
        if contacts:
            pos, quat = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=physics_client_id
            )
            contact_gt[label] = cast(SE3Pose, tuple(pos) + tuple(quat))

    if not contact_gt:
        return belief

    new_particles = []
    for particle in belief.particles:
        new_poses = dict(particle.object_poses)
        for label, gt_pose in contact_gt.items():
            particle_pose = new_poses.get(label)
            if particle_pose is not None:
                equiv_pose = _best_yaw_equivalent(
                    particle_pose, gt_pose, config.get_n_fold(label)
                )
                new_poses[label] = blend_poses(
                    particle_pose, equiv_pose, config.contact_alpha
                )
        new_particles.append(
            TabletopState(
                joint_positions=particle.joint_positions,
                gripper_open=particle.gripper_open,
                object_poses=new_poses,
            )
        )

    return Belief(
        particles=new_particles,
        weights=belief.weights.copy(),
        known_objects=belief.known_objects.copy(),
        occluded_objects=belief.occluded_objects.copy(),
        unknown_objects=belief.unknown_objects.copy(),
        object_confidence=dict(belief.object_confidence),
        held_object_label=belief.held_object_label,
        visibility_grid=belief.visibility_grid,
    )


def compute_belief_diagnostics(
    belief: Belief,
    camera_image,
    label_to_id: dict[str, int],
    physics_client_id: int,
    config: BeliefConfig | None = None,
) -> BeliefUpdateDiagnostics:
    """Compute diagnostics for belief update without modifying belief."""
    diagnostics = BeliefUpdateDiagnostics()
    config = config or BeliefConfig()
    detections = detect_objects_from_segmentation(
        camera_image.segmentation,
        label_to_id,
        physics_client_id,
        detection_pos_std=config.detection_pos_std,
        detection_distance_ref=config.detection_distance_ref,
    )

    weights = belief.weights.astype(np.float64)
    weights = weights / weights.sum()
    diagnostics.weight_entropy_before = float(
        -np.sum(weights * np.log(weights + 1e-10))
    )
    diagnostics.n_eff_before = float(1.0 / (weights**2).sum())
    diagnostics.weight_std_before = float(np.std(weights))
    diagnostics.weight_max_before = float(np.max(weights))

    # Per-object statistics
    for label, (detected_pose, _) in detections.items():
        distances = []
        likelihoods = []

        for particle in belief.particles:
            particle_pose = particle.object_poses[label]
            dist = pose_distance(particle_pose, detected_pose)
            distances.append(dist)
            likelihoods.append(
                float(np.exp(-0.5 * (dist / config.likelihood_distance_scale) ** 2))
            )

        stats = ObjectLikelihoodStats(
            label=label,
            mean_distance=float(np.mean(distances)),
            min_distance=float(np.min(distances)),
            max_distance=float(np.max(distances)),
            std_distance=float(np.std(distances)),
            mean_likelihood=float(np.mean(likelihoods)),
            min_likelihood=float(np.min(likelihoods)),
            max_likelihood=float(np.max(likelihoods)),
            total_particles=len(belief.particles),
        )
        diagnostics.object_stats[label] = stats

    # Compute what n_eff would be after update
    likelihoods_all = []
    for particle in belief.particles:
        likelihood = 1.0
        for label, (detected_pose, _) in detections.items():
            if label in belief.unknown_objects:
                continue
            particle_pose = particle.object_poses[label]
            dist = pose_distance(particle_pose, detected_pose)
            likelihood *= float(
                np.exp(-0.5 * (dist / config.likelihood_distance_scale) ** 2)
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
        label: stats.mean_distance for label, stats in diagnostics.object_stats.items()
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


def _is_position_visible(
    visibility_grid: LogOddsOccupancyGrid, position: tuple[float, ...]
) -> bool:
    """Check if position is in a visible (free) region of the grid."""
    return visibility_grid.is_free((position[0], position[1], position[2]))


def _fraction_visible_particles(
    belief: Belief, label: str, visibility_grid: LogOddsOccupancyGrid
) -> float:
    """Fraction of particles whose object pose lies in visible grid space."""
    poses = [
        particle.object_poses[label]
        for particle in belief.particles
        if label in particle.object_poses
    ]
    visible = sum(
        1 for pose in poses if _is_position_visible(visibility_grid, pose[:3])
    )
    return float(visible) / float(len(poses))


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

    all_labels: set[str] = set()
    for particle in belief.particles:
        all_labels.update(particle.object_poses.keys())

    mean_object_poses: dict[str, SE3Pose] = {}
    for label in all_labels:
        poses = [particle.object_poses[label] for particle in belief.particles]
        mean_object_poses[label] = average_poses(poses, belief.weights)

    return TabletopState(
        joint_positions=tuple(mean_joints),
        gripper_open=float(mean_gripper),
        object_poses=mean_object_poses,
    )


def get_mean_object_position(belief: Belief, label: str) -> np.ndarray | None:
    """Get weighted average position (xyz) for a specific object."""
    poses = [
        particle.object_poses[label]
        for particle in belief.particles
        if label in particle.object_poses
    ]
    positions_arr = np.array([[p[0], p[1], p[2]] for p in poses], dtype=np.float32)
    return np.average(positions_arr, axis=0, weights=belief.weights)


def add_pose_noise(pose: tuple[float, ...], pos_std: float, rot_std: float) -> SE3Pose:
    """Add Gaussian noise to SE(3) pose."""
    x, y, z, qx, qy, qz, qw = pose
    x += np.random.normal(0, pos_std)
    y += np.random.normal(0, pos_std)

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
    detections: dict[str, tuple[tuple[float, ...], float]],
    mean_distances: dict[str, float] | None = None,
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
    for label, (detected_pose, _) in detections.items():
        distances = [
            pose_distance(particle.object_poses[label], detected_pose)
            for particle in belief.particles
            if label in particle.object_poses
        ]
        weights_arr = belief.weights.astype(np.float64)
        weights_arr /= weights_arr.sum()
        mean_dist = float(np.sum(np.array(distances) * weights_arr))
        scores.append(float(np.exp(-0.5 * (mean_dist / distance_scale) ** 2)))

    if not scores:
        return 0.0

    return float(np.mean(scores))


def get_best_particle_state(belief: Belief) -> TabletopState:
    """Return the particle with the highest weight, or a random one if
    uniform."""
    weights = belief.weights
    if np.allclose(weights, weights[0]):
        return belief.particles[int(np.random.randint(len(belief.particles)))]
    return belief.particles[int(np.argmax(weights))]


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
