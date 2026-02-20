"""Tests for tabletop TAMP planning."""

import pytest

from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv
from residual_controllers.envs.tabletop_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    TabletopTypes,
    create_tabletop_operators,
    create_tabletop_skills,
)
from residual_controllers.tamp import PlanningComponents, run_tamp


@pytest.mark.skip(reason="Not complete.")
def test_tabletop_with_tamp():
    """Test TabletopPickEnv with bilevel TAMP planner."""
    seed = 123

    env = TabletopPickEnv(gui=False, num_objects=1, render_mode="rgb_array")
    sim = TabletopPickEnv(gui=False, num_objects=1)

    obs, _ = env.reset(seed=seed)
    sim.reset(seed=seed)

    sim.set_state(env.get_state())

    types = TabletopTypes()
    predicates = TabletopPredicates(types)
    operators = create_tabletop_operators(types, predicates)

    abstractor = TabletopAbstractor(sim, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(obs)
    print(f"Objects: {[o.name for o in objects]}")
    print(f"Initial atoms: {[str(a) for a in init_atoms]}")
    print(f"Goal atoms: {[str(a) for a in goal_atoms]}")

    components = PlanningComponents(
        types=types.as_set(),
        predicates=predicates,
        operators=operators,
        abstractor=abstractor,
    )

    skills = create_tabletop_skills(types, operators, sim, abstractor)

    def transition_fn(_obs_state, action):
        next_obs, _, _, _, _ = sim.step(action)
        return next_obs

    goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

    plan, graph = run_tamp(
        components=components,
        skills=skills,
        initial_state=obs,
        goal=goal,
        state_abstractor=abstractor.step,
        transition_function=transition_fn,
        timeout=30.0,
        seed=seed,
    )

    assert plan is not None, "Should find a plan"
    print("\nFound valid TAMP plan!")

    if graph.abstract_action_edges:
        abstract_actions = [action.name for _, action, _ in graph.abstract_action_edges]
        print(f"  - Abstract plan: {' → '.join(abstract_actions)}")

    print(f"  - Refined trajectory: {len(plan.actions)} low-level actions")

    print("\nExecuting plan on env...")
    for i, action in enumerate(plan.actions):
        gripper_action = action[7] if len(action) > 7 else 0.0

        if abs(gripper_action) > 0.5:
            action_type = "GRASP" if gripper_action < -0.5 else "RELEASE"
            print(f"\n  Step {i}: {action_type} action")
            print(
                f"    Before: env held={env._held_object_id}"  # pylint:disable=protected-access
            )

        obs, _, terminated, _, _ = env.step(action)

        if abs(gripper_action) > 0.5:
            belief = env.belief
            print(
                f"    After:  env held={env._held_object_id}"  # pylint:disable=protected-access
            )
            if belief is not None:
                print(f"    Belief: held_object_id={belief.held_object_id}")

                if belief.held_object_id is not None:
                    held_id = belief.held_object_id
                    poses = [p.object_poses.get(held_id) for p in belief.particles[:3]]
                    print("            First 3 particle poses for held obj:")
                    for j, pose in enumerate(poses):
                        if pose is not None:
                            print(
                                f"            [{j}]: ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})"  # pylint: disable=line-too-long
                            )

        if terminated:
            print(f"\n  Goal reached at step {i+1}!")
            break

    print("\n" + "=" * 50)
    print("Final belief state:")
    belief = env.belief
    if belief is not None:
        print(f"  held_object_id: {belief.held_object_id}")
        print(f"  known_objects: {belief.known_objects}")
        print(f"  unknown_objects: {belief.unknown_objects}")
    print("=" * 50)

    env.close()
    sim.close()
