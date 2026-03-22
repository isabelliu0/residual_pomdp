"""Collision utilities for arm-only collision checking.

Used to check arm link collisions with surfaces (e.g., table) while
avoiding false positives from the robot base sitting on the surface.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Callable, Optional

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    sample_collision_free_inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_joint_infos
from pybullet_helpers.math_utils import geometric_sequence
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    get_joint_positions_distance,
    get_motion_plan_distance,
    run_motion_planning,
)
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


def get_arm_link_ids(robot: SingleArmPyBulletRobot) -> list[int]:
    """Return all non-base link IDs (excludes base link -1)."""
    return list(
        range(p.getNumJoints(robot.robot_id, physicsClientId=robot.physics_client_id))
    )


def check_arm_body_collision(
    robot: SingleArmPyBulletRobot,
    body_id: int,
    distance_threshold: float = 1e-6,
) -> bool:
    """Check if any arm link (not base) is in collision with a body.

    Assumes p.performCollisionDetection has already been called.
    """
    for link_id in get_arm_link_ids(robot):
        if check_body_collisions(
            robot.robot_id,
            body_id,
            robot.physics_client_id,
            link1=link_id,
            distance_threshold=distance_threshold,
            perform_collision_detection=False,
        ):
            return True
    return False


def run_smooth_motion_planning_to_pose_with_surface_check(
    target_pose: Pose | Callable[[], Pose],
    robot: SingleArmPyBulletRobot,
    collision_ids: set[int],
    surface_id: int,
    end_effector_frame_to_plan_frame: Pose,
    seed: int,
    held_object: int | None = None,
    base_link_to_held_obj: Pose | None = None,
    max_time: float = np.inf,
    max_candidate_plans: int | None = None,
    joint_geometric_scalar: float = 0.9,
    birrt_num_attempts: int = 10,
    birrt_num_iters: int = 100,
    sampling_fn: Callable[[JointPositions], JointPositions] | None = None,
    distance_threshold: float = 1e-6,
) -> Optional[list[JointPositions]]:
    """Like run_smooth_motion_planning_to_pose but checks arm links vs surface.

    surface_id should be excluded from collision_ids to avoid false
    positives from the robot base resting on the surface. Arm links are
    still checked against surface_id via an additional constraint in
    run_motion_planning.
    """
    assert (
        not np.isinf(max_time) or max_candidate_plans is not None
    ), "Must specify either max_time or max_candidate_plans"

    if isinstance(target_pose, Pose):
        target_pose_sampler: Callable[[], Pose] = lambda: target_pose  # type: ignore
    else:
        target_pose_sampler = target_pose

    def _score_motion_plan(plan: list[JointPositions]) -> float:
        weights = geometric_sequence(joint_geometric_scalar, len(robot.arm_joints))
        joint_infos = get_joint_infos(
            robot.robot_id, robot.arm_joints, robot.physics_client_id
        )
        dist_fn = partial(
            get_joint_positions_distance,
            robot,
            joint_infos,
            metric="weighted_joints",
            weights=weights,
        )
        return get_motion_plan_distance(plan, dist_fn)

    def _arm_surface_ok(_joints: JointPositions) -> bool:
        return not check_arm_body_collision(robot, surface_id, distance_threshold)

    robot_initial_joints = robot.get_joint_positions()
    start_time = time.perf_counter()
    best_motion_plan: list[JointPositions] | None = None
    best_motion_plan_score: float = np.inf
    num_iters = 0
    iter_ub = max_candidate_plans if max_candidate_plans is not None else np.inf
    rng = np.random.default_rng(seed)

    while time.perf_counter() - start_time < max_time and num_iters < iter_ub:
        target_pose = target_pose_sampler()
        end_effector_pose = multiply_poses(
            target_pose, end_effector_frame_to_plan_frame
        )
        for target_joint_positions in sample_collision_free_inverse_kinematics(
            robot,
            end_effector_pose,
            collision_ids,
            rng,
            max_time=max_time,
            max_candidates=1,
        ):
            robot.set_joints(robot_initial_joints)
            motion_plan = run_motion_planning(
                robot,
                robot_initial_joints,
                target_joint_positions,
                collision_ids,
                seed,
                robot.physics_client_id,
                held_object=held_object,
                base_link_to_held_obj=base_link_to_held_obj,
                sampling_fn=sampling_fn,
                hyperparameters=MotionPlanningHyperparameters(
                    birrt_num_attempts=birrt_num_attempts,
                    birrt_num_iters=birrt_num_iters,
                ),
                additional_state_constraint_fn=_arm_surface_ok,
                distance_threshold=distance_threshold,
            )
            if motion_plan is not None:
                num_iters += 1
                score = _score_motion_plan(motion_plan)
                if score < best_motion_plan_score:
                    best_motion_plan = motion_plan
                    best_motion_plan_score = score

    return best_motion_plan
