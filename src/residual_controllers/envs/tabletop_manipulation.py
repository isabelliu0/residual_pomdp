"""Utilities for object manipulation."""

from typing import Iterator

import numpy as np
from pybullet_helpers.geometry import (
    Pose,
    get_half_extents_from_aabb,
    iter_between_poses,
    multiply_poses,
)
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.states import KinematicState
from pybullet_helpers.utils import get_closest_points_with_optional_links


def get_kinematic_plan_to_pick_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    grasp_generator: Iterator[Pose],
    grasp_generator_iters: int = int(1e6),
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    pregrasp_pad_scale: float = 1.1,
    postgrasp_translation: Pose | None = None,
    postgrasp_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_motion_planning_candidates: int | None = None,
    max_smoothing_iters_per_step: int = 1,
    seed: int = 0,
) -> list[KinematicState] | None:
    """Make a plan to pick up the object from a surface.

    The grasp pose is in the object frame.

    The surface is used to determine the direction that the robot should move
    directly after picking (to remove contact between the object and surface).

    Users should make grasp_generator finite to prevent infinite loops, unless
    they are very confident that some feasible grasp plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    all_object_ids = set(state.object_poses)
    joint_distance_fn = create_joint_distance_fn(robot)

    # Calculate pregrasp poses by translating the grasp away from the object.
    # The translation amount is determined based on the size of the axis aligned
    # bounding box for the object and the robot end effector.
    pregrasp_distance = _get_approach_distance_from_aabbs(
        robot, object_id, object_link_id=object_link_id, pad_scale=pregrasp_pad_scale
    )

    # Calculate once the direction to move after grasping succeeds. Using the
    # contact normal with the surface.
    if postgrasp_translation is None:
        postgrasp_translation = _get_approach_pose_from_contact_normals(
            object_id,
            surface_id,
            robot.physics_client_id,
            surface_link_id=surface_link_id,
            translation_magnitude=postgrasp_translation_magnitude,
        )

    # Prepare to transform grasps relative to the link into the object frame.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )

    num_attempts = 0
    for relative_grasp in grasp_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the grasp in the world frame.
        object_pose = state.object_poses[object_id]
        grasp = multiply_poses(object_pose, object_to_link, relative_grasp)

        # Calculate the pregrasp pose.
        pregrasp_translation_direction = np.array([0.0, 0.0, -1.0])
        pregrasp_tf = Pose(tuple(pregrasp_translation_direction * pregrasp_distance))
        pregrasp_pose = multiply_poses(grasp, pregrasp_tf)

        # Motion plan to the pregrasp pose.
        plan_to_pregrasp = run_smooth_motion_planning_to_pose(
            pregrasp_pose,
            robot,
            collision_ids=collision_ids - {surface_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
        )
        num_attempts += 1
        # If motion planning failed, try a different grasp.
        if plan_to_pregrasp is None:
            if num_attempts >= grasp_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_pregrasp:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Move to grasp.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                grasp,
                include_start=False,
            )
        )
        try:
            pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                include_start=False,
            )
        except InverseKinematicsError:
            pregrasp_to_grasp_plan = None
        # If motion planning failed, try a different grasp.
        if pregrasp_to_grasp_plan is None:
            if num_attempts >= grasp_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in pregrasp_to_grasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Update the state to include a grasp attachment.
        state = KinematicState.from_pybullet(
            robot, all_object_ids, attached_object_ids={object_id}
        )
        plan.append(state)

        # Move off the surface.
        end_effector_pose = robot.get_end_effector_pose()
        post_grasp_pose = multiply_poses(postgrasp_translation, end_effector_pose)
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                post_grasp_pose,
                include_start=False,
            )
        )

        try:
            grasp_to_postgrasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                include_start=False,
                held_object=object_id,
                base_link_to_held_obj=relative_grasp.invert(),
            )
        except InverseKinematicsError:
            grasp_to_postgrasp_plan = None
        # If motion planning failed, try a different grasp.
        if grasp_to_postgrasp_plan is None:
            if num_attempts >= grasp_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in grasp_to_postgrasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        # Planning succeeded.
        return plan

    # No grasp worked.
    return None


def _get_approach_distance_from_aabbs(
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    object_link_id: int | None = None,
    pad_scale: float = 1.1,
) -> float:
    object_half_extents = get_half_extents_from_aabb(
        object_id,
        physics_client_id=robot.physics_client_id,
        link_id=object_link_id,
        rotation_okay=True,
    )
    object_radius = max(object_half_extents)
    robot_end_effector_radius = 0.0  # find max value over fingers
    for finger_id in robot.finger_ids:
        robot_end_effector_half_extents = get_half_extents_from_aabb(
            robot.robot_id,
            physics_client_id=robot.physics_client_id,
            link_id=finger_id,
            rotation_okay=True,
        )
        robot_end_effector_radius = max(
            robot_end_effector_radius,
            *robot_end_effector_half_extents,
        )

    return (object_radius + robot_end_effector_radius) * pad_scale


def _get_approach_pose_from_contact_normals(
    object_id: int,
    surface_id: int,
    physics_client_id: int,
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    translation_magnitude: float = 0.05,
    contact_distance_threshold: float = 1e-3,
):
    contact_points = get_closest_points_with_optional_links(
        object_id,
        surface_id,
        physics_client_id=physics_client_id,
        link1=object_link_id,
        link2=surface_link_id,
        distance_threshold=contact_distance_threshold,
    )
    assert len(contact_points) > 0
    contact_normals = []
    for contact_point in contact_points:
        contact_normal = contact_point[7]
        contact_normals.append(contact_normal)
    vec = np.mean(contact_normals, axis=0)
    translation_direction = vec / np.linalg.norm(vec)
    translation = translation_direction * translation_magnitude
    return Pose(tuple(translation))
