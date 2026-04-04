"""Tests for kinematic planning edge cases."""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
)

from residual_controllers.envs.tabletop_tamp_base import (
    PickGroundController,
    TabletopTypes,
)
from residual_controllers.envs.tabletop_view_occlusion import TabletopViewOcclusionEnv
from residual_controllers.envs.tabletop_view_occlusion_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
)


def test_pick_planning_succeeds_from_gripper_contact() -> None:
    """Pick planning succeeds even when the gripper starts in contact with
    target.

    Simulates the RL-contact scenario: a manual side-approach has already moved one
    finger into contact with the target object, and now TAMP must replan a Pick
    from that contact state. With make_constraint_with_start_skip, planning should
    succeed despite the initial contact.
    """
    env = TabletopViewOcclusionEnv(gui=False, num_objects=1)
    try:
        env.reset(seed=42)

        types = TabletopTypes()
        predicates = TabletopPredicates(types)
        abstractor = TabletopAbstractor(env, types, predicates)
        gt_obs = env.get_observation()
        abstractor.reset(gt_obs)

        robot_obj = abstractor._robot_obj  # pylint: disable=protected-access
        movable_obj = abstractor._movable_objs[0]  # pylint: disable=protected-access
        table_obj = abstractor._table_obj  # pylint: disable=protected-access
        obj_id = abstractor.get_pybullet_id(movable_obj)

        assert env.robot is not None

        obj_pos, _ = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=env.physics_client_id
        )
        ox, oy, oz = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])

        aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=env.physics_client_id)
        obj_half_y = 0.5 * (float(aabb_max[1]) - float(aabb_min[1]))
        ee_z = oz + 0.04
        approach_y = oy + obj_half_y + 0.12
        contact_y = oy + obj_half_y - 0.05

        def _finger_contact_points(distance: float = 2e-3) -> list[tuple]:
            p.performCollisionDetection(physicsClientId=env.physics_client_id)
            return [
                pt
                for pt in p.getClosestPoints(
                    env.robot.robot_id,
                    obj_id,
                    distance=distance,
                    physicsClientId=env.physics_client_id,
                )
                if pt[3] in env.robot.finger_ids
            ]

        def _move_ee_to(
            target_pos: tuple[float, float, float], n_steps: int = 10
        ) -> None:
            for _ in range(n_steps):
                ik = p.calculateInverseKinematics(
                    env.robot.robot_id,
                    env.robot.end_effector_id,
                    target_pos,
                    maxNumIterations=1000,
                    residualThreshold=1e-6,
                    physicsClientId=env.physics_client_id,
                )
                current = np.array(
                    env.robot.get_joint_positions()[:7], dtype=np.float32
                )
                target = np.array(ik[:7], dtype=np.float32)
                joint_delta = np.clip(target - current, -0.03, 0.03)
                env.step(np.concatenate([joint_delta, [0.0]]).astype(np.float32))

        _move_ee_to((ox, approach_y, ee_z), n_steps=25)
        for y in np.linspace(approach_y, contact_y, num=10):
            _move_ee_to((ox, float(y), ee_z), n_steps=4)
            if any(float(pt[8]) <= 1e-3 for pt in _finger_contact_points()):
                break

        in_contact = check_body_collisions(
            env.robot.robot_id,
            obj_id,
            env.physics_client_id,
            distance_threshold=1e-3,
        )
        finger_in_contact = any(float(pt[8]) <= 1e-3 for pt in _finger_contact_points())
        if not (in_contact and finger_in_contact):
            pytest.skip(
                "Could not establish finger-object side contact for this configuration"
            )

        obs = env.get_observation()

        # Planning from the contact state must succeed (not raise
        # TrajectorySamplingFailure) thanks to start_ignored_collision_bodies.
        controller = PickGroundController(
            [robot_obj, movable_obj, table_obj], env, abstractor
        )
        controller.reset(obs, {})

    finally:
        env.close()
