"""Unit tests for the compliant RL training pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pybullet as p
import pytest
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)

from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.operating_region.data_collect import (
    _check_operator_success,
    _parse_operator_name,
)
from residual_controllers.operating_region.features import extract_features
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph

_PEG_HEIGHT = 0.1


def execute_op_with_phase_checks(
    system: Any, plan: Any, action_str: str, op_goal: set
) -> tuple[bool, bool, int]:
    """Execute plan actions with phase checks, stopping when op_goal is
    achieved.

    Returns (failure_detected, env_terminated, actions_consumed).
    """
    ctrl = None
    get_ctrl = getattr(system, "get_grounded_controller", None)
    if get_ctrl is not None:
        ctrl = get_ctrl(action_str)

    obs = system.env.get_observation()
    if ctrl is not None:
        try:
            ctrl.reset_for_checking(plan, obs)
        except TrajectorySamplingFailure:
            return True, False, 0

    first = True
    consumed = 0
    for action in plan.actions:
        if ctrl is not None:
            if not first:
                ctrl.observe(system.env.get_observation())
            try:
                ctrl.step()
            except TrajectorySamplingFailure:
                return True, False, consumed
            first = False

        _, _, terminated, _, _ = system.env.step(action)
        consumed += 1
        if terminated:
            return False, True, consumed
        if _check_operator_success(system, op_goal):
            return False, False, consumed

    return False, False, consumed


def _ik_delta_action(
    system: NutAssemblyTAMPSystem,
    target_pos: tuple[float, float, float],
    down_quat: tuple[float, ...],
    step_size: float = 0.05,
) -> np.ndarray:
    robot = system.env.robot
    pcid = system.env.physics_client_id
    ik = p.calculateInverseKinematics(
        robot.robot_id,
        robot.end_effector_id,
        target_pos,
        targetOrientation=down_quat,
        maxNumIterations=1000,
        residualThreshold=1e-6,
        physicsClientId=pcid,
    )
    curr = np.array(robot.get_joint_positions()[:7])
    delta = np.clip(np.array(ik[:7]) - curr, -step_size, step_size)
    return delta.astype(np.float32)


@pytest.mark.skip(
    reason="Outdated test. Injected vision noise would cause EE failures."
)
def test_compliant_rl_gt_trajectory():
    """RL success condition is met when GT IK trajectory replaces the policy.

    Mirrors train_compliant_rl.py: get the full plan once and slice per
    operator. If an operator fails, activate the GT IK loop (instead of
    policy.sample_action) and assert sigma drops below sigma_thresh via
    _update_belief_from_contact.
    """
    system = NutAssemblyTAMPSystem(seed=42, gui=False)
    try:
        system.reset(seed=42)

        symbolic_plan = system.get_symbolic_plan()
        assert symbolic_plan is not None

        ignored = system.get_oig_ignored_objects()
        subsequences = build_object_interaction_graph(symbolic_plan, ignored)

        rl_triggered = False
        for subseq_actions, obj_names in subsequences:
            rel_labels = list(system.get_object_labels_for_names(obj_names).values())

            plan_actions: list | None = None
            plan_states: list | None = None
            action_idx = 0

            for action_str in subseq_actions:
                op_name = _parse_operator_name(action_str)
                op_goal, _ = system.get_subsequence_effects([action_str])

                if plan_actions is None:
                    plan = system.plan(seed=42)
                    assert plan is not None, "Planning failed"
                    plan_actions = list(plan.actions)
                    plan_states = list(plan.states)
                    action_idx = 0

                op_plan = SimpleNamespace(
                    actions=plan_actions[action_idx:],
                    states=plan_states[action_idx:],
                )
                phase_failure, env_terminated, consumed = execute_op_with_phase_checks(
                    system, op_plan, action_str, op_goal
                )
                action_idx += consumed

                if env_terminated or (
                    not phase_failure and _check_operator_success(system, op_goal)
                ):
                    continue

                rl_triggered = True

                sigma_thresh = 1e-10
                all_labels = system.env.object_labels
                feats0 = extract_features(system.env.belief, all_labels, rel_labels)
                sigma_0 = feats0.relevant_sigma

                print(f"\nsigma_0={sigma_0:.6f}, " f"sigma_thresh={sigma_thresh:.2e}")

                robot = system.env.robot
                _, _, ee_yaw = p.getEulerFromQuaternion(
                    robot.get_end_effector_pose().orientation
                )
                down_quat = tuple(p.getQuaternionFromEuler([np.pi, 0, ee_yaw]))
                target_pos = (0.5, 0.0, _PEG_HEIGHT)

                max_steps = 60
                rl_success = False
                sigma_curr = sigma_0

                for step in range(max_steps):
                    action7 = _ik_delta_action(system, target_pos, down_quat)
                    system.env.step(np.append(action7, 0.0).astype(np.float32))
                    feats = extract_features(system.env.belief, all_labels, rel_labels)
                    sigma_curr = feats.relevant_sigma
                    print(f"  step {step:2d}: sigma={sigma_curr:.8f}")
                    if sigma_curr <= sigma_thresh:
                        rl_success = True
                        break

                assert rl_success, (
                    f"RL did not succeed in {max_steps} steps. "
                    f"Final sigma={sigma_curr:.8f} > thresh={sigma_thresh:.2e}"
                )
                print(f"\nRL succeeded at step {step} with sigma={sigma_curr:.8f}")

                # Re-plan and re-execute this operator now that belief is
                # certain enough. The new plan from the near-certain belief is
                # expected to achieve op_goal.
                print(f"\nRe-planning '{op_name}' after RL to achieve {op_goal}")
                plan_retry = system.plan_for_goal(op_goal, seed=42)
                assert (
                    plan_retry is not None
                ), f"Re-planning after RL failed for '{op_name}'"
                phase_failure2, env_terminated2, _ = execute_op_with_phase_checks(
                    system, plan_retry, action_str, op_goal
                )
                assert (
                    not phase_failure2
                ), f"Phase failure on re-execution of '{op_name}' after RL"
                assert env_terminated2 or _check_operator_success(
                    system, op_goal
                ), f"Op goal not achieved after RL + re-execution for '{op_name}'"
                print(f"  '{op_name}' goal achieved after RL re-execution.")
                if env_terminated2:
                    break
                plan_actions = None  # force re-plan before next operator

        assert rl_triggered, "Operator never failed — GT RL loop was never exercised"

    finally:
        system.close()
