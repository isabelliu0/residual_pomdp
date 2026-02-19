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


@pytest.mark.skip(reason="TODO: Add grasp transform")
def test_tabletop_with_tamp():
    """Test TabletopPickEnv with bilevel TAMP planner."""
    env = TabletopPickEnv(gui=True, num_objects=1)
    obs, _ = env.reset(seed=123)

    types = TabletopTypes()
    predicates = TabletopPredicates(types)
    operators = create_tabletop_operators(types, predicates)
    abstractor = TabletopAbstractor(env, types, predicates)

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

    skills = create_tabletop_skills(types, operators, env, abstractor)

    def transition_fn(_obs_state, action):
        next_obs, _, _, _, _ = env.step(action)
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
        seed=123,
    )

    assert plan is not None, "Should find a plan"
    print("\nFound valid TAMP plan!")

    if graph.abstract_action_edges:
        abstract_actions = [action.name for _, action, _ in graph.abstract_action_edges]
        print(f"  - Abstract plan: {' → '.join(abstract_actions)}")

    print(f"  - Refined trajectory: {len(plan.actions)} low-level actions")

    env.close()
