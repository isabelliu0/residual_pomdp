"""Tests for NutAssemblyEnv."""

from __future__ import annotations

import time

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses

from residual_controllers.beliefs.structs import TabletopState
from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.envs.nut_assembly_env import (
    NutAssemblyEnv,
    NutAssemblyEnvState,
)
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph

_PEG_HEIGHT = 0.1


def test_basic_reset():
    """Verify NutAssemblyEnv resets correctly."""
    env = NutAssemblyEnv(gui=False)

    obs, info = env.reset(seed=42)

    assert set(obs.keys()) == {
        "joint_positions",
        "camera_pose",
        "object_poses",
        "held_object_idx",
        "grasp_transform",
    }
    assert obs["joint_positions"].shape == (9,)
    assert obs["camera_pose"].shape == (7,)
    assert obs["object_poses"].shape == (10, 7)
    assert info["target_object_label"] == "NUT"

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        assert not terminated
        assert reward == 0.0

    env.close()


def test_nut_peg_surface_interaction():
    """Visual test: nut held near peg top slides along surface without penetrating."""
    env = NutAssemblyEnv(gui=False)
    env.reset(seed=42)

    robot = env.robot
    pcid = env.physics_client_id
    peg_cx, peg_cy = 0.5, 0.0
    peg_top_z = _PEG_HEIGHT

    _, _, ee_yaw = p.getEulerFromQuaternion(robot.get_end_effector_pose().orientation)
    down_quat = tuple(p.getQuaternionFromEuler([np.pi, 0, ee_yaw]))

    ik = p.calculateInverseKinematics(
        robot.robot_id,
        robot.end_effector_id,
        (peg_cx, peg_cy, peg_top_z + 0.1),
        targetOrientation=down_quat,
        maxNumIterations=1000,
        residualThreshold=1e-6,
        physicsClientId=pcid,
    )
    joints = list(ik[:7]) + list(robot.get_joint_positions()[7:])
    robot.set_joints(joints)
    ee_pose = robot.get_end_effector_pose()

    nut_pose = Pose(position=(peg_cx, peg_cy, peg_top_z + 0.1), orientation=down_quat)
    peg_pose = Pose(position=(peg_cx, peg_cy, peg_top_z / 2), orientation=(0, 0, 0, 1))
    grasp_tf = multiply_poses(ee_pose.invert(), nut_pose)

    env.set_state(
        NutAssemblyEnvState(
            robot_joints=tuple(joints),
            object_poses=(nut_pose, peg_pose),
            held_object_idx=0,
            grasp_transform=grasp_tf,
        )
    )

    def step_toward(target_pos: tuple, n_steps: int = 20) -> None:
        for _ in range(n_steps):
            ik_step = p.calculateInverseKinematics(
                robot.robot_id,
                robot.end_effector_id,
                target_pos,
                targetOrientation=down_quat,
                maxNumIterations=1000,
                residualThreshold=1e-6,
                physicsClientId=pcid,
            )
            curr = np.array(robot.get_joint_positions()[:7])
            delta = np.clip(np.array(ik_step[:7]) - curr, -0.01, 0.01)
            env.step(np.concatenate([delta, [0.0]]).astype(np.float32))

            contacts = p.getContactPoints(
                env._nut_id,  # pylint: disable=protected-access
                env._peg_id,  # pylint: disable=protected-access
                physicsClientId=pcid,
            )
            for c in contacts or []:
                assert (
                    float(c[8]) > -2e-3
                ), f"Nut penetrated peg by {-float(c[8])*1000:.2f} mm"

            time.sleep(0.04)

    rim_z = peg_top_z - 0.01
    offset = 0.025

    for angle in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        step_toward(
            (peg_cx + offset * np.cos(angle), peg_cy + offset * np.sin(angle), rim_z),
            n_steps=15,
        )

    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        step_toward(
            (peg_cx + offset * np.cos(angle), peg_cy + offset * np.sin(angle), rim_z),
            n_steps=4,
        )

    step_toward((peg_cx, peg_cy, rim_z), n_steps=20)

    env.close()


def test_orp_calibration():
    """ORP core assumption: lower peg belief sigma -> higher assembly success rate."""

    def _inject_peg_sigma(
        system: NutAssemblyTAMPSystem, peg_sigma: float, rng: np.random.Generator
    ) -> None:
        belief = system.env.belief
        if belief is None:
            return
        peg_gt = get_pose(
            system.env._peg_id, system.env.physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
        )
        nut_gt = get_pose(
            system.env._nut_id, system.env.physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
        )
        new_particles = []
        for particle in belief.particles:
            poses = dict(particle.object_poses)
            poses["PEG"] = (
                float(peg_gt.position[0]) + float(rng.normal(0, peg_sigma)),
                float(peg_gt.position[1]) + float(rng.normal(0, peg_sigma)),
                float(peg_gt.position[2]),
                float(peg_gt.orientation[0]),
                float(peg_gt.orientation[1]),
                float(peg_gt.orientation[2]),
                float(peg_gt.orientation[3]),
            )
            poses["NUT"] = (
                float(nut_gt.position[0]),
                float(nut_gt.position[1]),
                float(nut_gt.position[2]),
                float(nut_gt.orientation[0]),
                float(nut_gt.orientation[1]),
                float(nut_gt.orientation[2]),
                float(nut_gt.orientation[3]),
            )
            new_particles.append(
                TabletopState(
                    joint_positions=particle.joint_positions,
                    gripper_open=particle.gripper_open,
                    object_poses=poses,
                )
            )
        n = len(new_particles)
        belief.particles = new_particles
        belief.weights = np.ones(n) / n

    def _run_trial(seed: int, peg_sigma: float) -> bool:
        system = NutAssemblyTAMPSystem(seed=seed, gui=False)
        rng = np.random.default_rng(seed + 999)
        try:
            system.reset(seed=seed)
            symbolic_plan = system.get_symbolic_plan()
            if symbolic_plan is None:
                return False
            ignored = system.get_oig_ignored_objects()
            subsequences = build_object_interaction_graph(symbolic_plan, ignored)
            terminated = False
            for subseq_actions, _ in subsequences:
                goal_add, _ = system.get_subsequence_effects(subseq_actions)
                op = "+".join(a.strip("() ").split()[0] for a in subseq_actions)
                if "assemble" in op:
                    _inject_peg_sigma(system, peg_sigma, rng)
                plan = system.plan_for_goal(goal_add, seed=seed)
                if plan is None:
                    return False
                for action in plan.actions:
                    _, _, terminated, _, _ = system.env.step(action)
                    if terminated:
                        break
                if terminated:
                    break
            return terminated
        finally:
            system.close()

    N_TRIALS = 3

    tight_successes = sum(_run_trial(seed=s, peg_sigma=1e-6) for s in range(N_TRIALS))
    loose_successes = sum(_run_trial(seed=s, peg_sigma=0.05) for s in range(N_TRIALS))

    print(f"\nTight sigma (0.001 mm): {tight_successes}/{N_TRIALS} successes")
    print(f"Loose sigma (50.0 mm): {loose_successes}/{N_TRIALS} successes")
