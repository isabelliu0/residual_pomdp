"""Utility functions for residual controllers on Cover2DEnv."""

from __future__ import annotations

import numpy as np

from residual_controllers.envs.cover2d import Action, Belief


def encode_belief_cover2d(belief: Belief) -> np.ndarray:
    """Encode Cover2DEnv belief into a fixed-size observation vector."""
    robot_x_mean = np.mean([p.robot_pose.x for p in belief.particles])
    robot_y_mean = np.mean([p.robot_pose.y for p in belief.particles])
    robot_theta_mean = np.mean([p.robot_pose.theta for p in belief.particles])

    robot_x_std = np.std([p.robot_pose.x for p in belief.particles])
    robot_y_std = np.std([p.robot_pose.y for p in belief.particles])
    robot_theta_std = np.std([p.robot_pose.theta for p in belief.particles])

    best_particle_idx = np.argmax(belief.weights)
    best_particle = belief.particles[best_particle_idx]
    is_holding = float(best_particle.gripper_state.is_holding)

    if best_particle.object_poses:
        obj_x_mean = float(
            np.mean(
                [
                    list(p.object_poses.values())[0].x
                    for p in belief.particles
                    if p.object_poses
                ]
            )
        )
        obj_y_mean = float(
            np.mean(
                [
                    list(p.object_poses.values())[0].y
                    for p in belief.particles
                    if p.object_poses
                ]
            )
        )
    else:
        obj_x_mean = 0.0
        obj_y_mean = 0.0

    return np.array(
        [
            robot_x_mean,
            robot_y_mean,
            robot_theta_mean,
            robot_x_std,
            robot_y_std,
            robot_theta_std,
            is_holding,
            obj_x_mean,
            obj_y_mean,
        ],
        dtype=np.float32,
    )


def action_from_residual(base_action: Action, residual: np.ndarray) -> Action:
    """Create a new action by adding residual corrections to the base
    action."""
    return Action(
        dx=base_action.dx + residual[0],
        dy=base_action.dy + residual[1],
        dtheta=base_action.dtheta,
        gripper_action=base_action.gripper_action,
    )
