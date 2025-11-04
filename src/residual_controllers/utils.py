"""Utilities."""

from typing import Any

import numpy as np
from tampura_environments.slam_collect.env import Pose


def select_mean_particle(belief: Any) -> Any:
    """Select a canonical particle deterministically from belief."""
    if not hasattr(belief, "particles") or len(belief.particles) == 0:
        return belief
    particles = belief.particles
    sorted_particles = sorted(
        particles, key=lambda p: (p.robot.pose.x, p.robot.pose.y, p.robot.pose.theta)
    )
    mean_particle = sorted_particles[0]
    mean_particle.robot.pose = Pose.from_array(
        np.mean(np.array([p.robot.pose.to_array() for p in sorted_particles]), axis=0)
    )
    return mean_particle


def encode_slam_belief(belief: Any) -> np.ndarray:
    """Encode a SLAM belief state as a fixed-size numpy array."""
    if not hasattr(belief, "particles") or len(belief.particles) == 0:
        raise ValueError("Belief must have non-empty particles attribute")

    particles = belief.particles
    first_particle = particles[0]
    feature_parts = []

    robot_poses = np.array([p.robot.pose.to_array() for p in particles])
    robot_mean = np.mean(robot_poses, axis=0)  # [x, y, theta]
    robot_std = np.std(robot_poses, axis=0)
    feature_parts.extend([robot_mean, robot_std])

    if hasattr(first_particle, "targets") and len(first_particle.targets) > 0:
        target_poses = np.array(
            [[t.pose.to_array() for t in p.targets] for p in particles]
        )  # Shape: (num_particles, num_targets, 3)
        target_mean = np.mean(target_poses, axis=0).flatten()  # (num_targets * 3,)
        target_std = np.std(target_poses, axis=0).flatten()
        feature_parts.extend([target_mean, target_std])

    # Static environment features
    if hasattr(first_particle, "obstacles") and len(first_particle.obstacles) > 0:
        obstacle_poses = np.array(
            [obs.pose.to_array() for obs in first_particle.obstacles]
        ).flatten()
        feature_parts.append(obstacle_poses)

    if hasattr(first_particle, "beacons") and len(first_particle.beacons) > 0:
        beacon_poses = np.array(
            [beacon.pose.to_array() for beacon in first_particle.beacons]
        ).flatten()
        feature_parts.append(beacon_poses)

    if hasattr(first_particle, "corners") and len(first_particle.corners) > 0:
        corner_poses = np.array(
            [corner.pose.to_array() for corner in first_particle.corners]
        ).flatten()
        feature_parts.append(corner_poses)

    if hasattr(first_particle, "goal") and first_particle.goal is not None:
        goal_pose = first_particle.goal.pose.to_array()
        feature_parts.append(goal_pose)

    encoded = np.concatenate(feature_parts)
    return encoded.astype(np.float32)
