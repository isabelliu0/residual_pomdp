"""Tests for tabletop TAMP planning."""

import pytest
from gymnasium.wrappers import RecordVideo

from residual_controllers.beliefs import get_mean_state
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv
from residual_controllers.envs.tabletop_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    TabletopTypes,
    create_tabletop_operators,
    create_tabletop_skills,
)
from residual_controllers.tamp import PlanningComponents, run_tamp


@pytest.mark.skip()
def test_tabletop_with_tamp():
    """Test TabletopPickEnv pick-and-place with bilevel TAMP planner."""
    seed = 123

    base_env = TabletopPickEnv(
        gui=False, num_objects=1, render_mode="rgb_array_external"
    )
    env = RecordVideo(base_env, "videos/planning")
    sim = TabletopPickEnv(gui=False, num_objects=1)

    _, _ = env.reset(seed=seed)
    sim.reset(seed=seed)
    sim.set_state(env.get_state())

    assert base_env.belief is not None
    belief_obs = sim.get_obs_from_mean(get_mean_state(base_env.belief))

    types = TabletopTypes()
    predicates = TabletopPredicates(types)
    operators = create_tabletop_operators(types, predicates)

    abstractor = TabletopAbstractor(sim, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(belief_obs)
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
        initial_state=belief_obs,
        goal=goal,
        state_abstractor=abstractor.step,
        transition_function=transition_fn,
        timeout=60.0,
        seed=seed,
    )

    assert plan is not None, "Should find a plan"
    print("\nFound valid TAMP plan!")

    if graph.abstract_action_edges:
        abstract_actions = [action.name for _, action, _ in graph.abstract_action_edges]
        print(f"  - Abstract plan: {' -> '.join(abstract_actions)}")

    print(f"  - Refined trajectory: {len(plan.actions)} low-level actions")

    print("\nExecuting plan on env...")
    terminated = False
    for i, action in enumerate(plan.actions):
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            print(f"\n  Goal reached at step {i+1}!")
            break

    print("\n" + "=" * 50)
    assert terminated, "Should reach goal (target placed in target area)"
    print("=" * 50)

    env.close()
    sim.close()
