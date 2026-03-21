"""Tests for operating region predictor."""

from __future__ import annotations

import pytest
from pybullet_helpers.geometry import Pose

from residual_controllers.benchmarks import TabletopPickTAMPSystem
from residual_controllers.envs.tabletop_pybullet import TabletopEnvState
from residual_controllers.operating_region.data_collect import collect_episode
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph


@pytest.mark.skip(reason="Requires GUI.")
def test_collect_episode() -> None:
    """Collect episodes with GUI to inspect data collection."""
    n_episodes = 5

    all_records = []
    for ep in range(n_episodes):
        system = TabletopPickTAMPSystem(seed=ep + 42, gui=True, num_objects=1)
        records = collect_episode(system, seed=ep + 42, nbv_step_counts=(0, 10, 20))
        all_records.extend(records)
        print(f"\n=== Episode {ep}: {len(records)} records ===")
        for r in records:
            print(
                f"  [{r.source:12s}] op={r.operator_name}"
                f"  relevant_sigma={r.features.relevant_sigma:.4f}"
                f"  success={r.success}"
                f"  meta={r.metadata}"
            )
        system.close()

    print(f"\n=== Total: {len(all_records)} records across {n_episodes} episodes ===")
    assert len(all_records) > 0


@pytest.mark.skip(reason="Takes too long to run.")
def test_particle_rollout_ablation() -> None:
    """Ablation: evaluate robustness of the MAP plan across belief particles.

    Computes one plan from the MAP (mean) belief, then executes that fixed plan
    in the planning env reset to each particle's poses. This measures how often
    the plan the agent would actually execute succeeds across the belief distribution.

    P(success) = fraction of particles where the fixed MAP plan achieves the goal.
    If below threshold, NBV would be triggered to reduce uncertainty first.
    """
    system = TabletopPickTAMPSystem(seed=42, gui=False, num_objects=5)
    system.reset(seed=42)

    symbolic_plan = system.get_symbolic_plan()
    if symbolic_plan is None:
        system.close()
        pytest.skip("No symbolic plan found")

    ignored = system.get_oig_ignored_objects()
    subsequences = build_object_interaction_graph(symbolic_plan, ignored)
    if not subsequences:
        system.close()
        pytest.skip("No OIG subsequences")

    subseq_actions, _ = subsequences[0]
    goal_add, _ = system.get_subsequence_effects(subseq_actions)
    all_obj_ids = list(system.env.scene.object_ids)

    belief = system.env.belief
    if belief is None or not belief.particles:
        system.close()
        pytest.skip("No belief particles")

    real_env_state = system.env.get_state()

    map_plan = system.plan_for_goal(goal_add, seed=42)

    assert system._plan_env is not None  # type: ignore[attr-defined]   # pylint: disable=protected-access
    plan_env = system._plan_env  # type: ignore[attr-defined]   # pylint: disable=protected-access

    n_particles = min(5, len(belief.particles))
    successes = 0

    for i, particle in enumerate(belief.particles[:n_particles]):
        particle_poses = tuple(
            Pose(
                position=particle.object_poses[obj_id][:3],
                orientation=particle.object_poses[obj_id][3:],
            )
            for obj_id in all_obj_ids
        )
        plan_env.set_state(
            TabletopEnvState(
                robot_joints=real_env_state.robot_joints,
                object_poses=particle_poses,
                held_object_idx=real_env_state.held_object_idx,
                grasp_transform=real_env_state.grasp_transform,
            )
        )

        if map_plan is not None:
            for action in map_plan.actions:
                plan_env.step(action)
            obs = plan_env.get_observation()
            assert system.abstractor is not None
            abstract_state = system.abstractor.step(obs)
            success = goal_add.issubset(abstract_state.atoms)
        else:
            success = False

        successes += int(success)
        print(f"  [particle {i + 1}/{n_particles}] success={success}")

    p_success = successes / n_particles
    threshold = 0.8
    nbv_would_trigger = p_success < threshold

    print("\n=== Particle Rollout Ablation ===")
    print(f"  Particles sampled: {n_particles}")
    print(f"  Successes:         {successes}")
    print(f"  P(success):        {p_success:.2f}")
    print(f"  Threshold:         {threshold}")
    print(f"  NBV would trigger: {nbv_would_trigger}")

    assert 0.0 <= p_success <= 1.0

    system.close()
