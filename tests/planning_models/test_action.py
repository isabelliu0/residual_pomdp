"""Tests for tabletop TAMP planning."""

from gymnasium.wrappers import RecordVideo

from residual_controllers.envs.nut_assembly_env import NutAssemblyEnv
from residual_controllers.envs.nut_assembly_tamp import (
    NutAssemblyAbstractor,
    NutAssemblyPredicates,
    create_nut_assembly_operators,
    create_nut_assembly_skills,
)
from residual_controllers.envs.tabletop_object_occlusion import (
    TabletopObjectOcclusionEnv,
)
from residual_controllers.envs.tabletop_object_occlusion_tamp import (
    TabletopPourAbstractor,
    TabletopPourPredicates,
    create_tabletop_pour_operators,
    create_tabletop_pour_skills,
)
from residual_controllers.envs.tabletop_tamp_base import TabletopTypes
from residual_controllers.envs.tabletop_view_occlusion import TabletopViewOcclusionEnv
from residual_controllers.envs.tabletop_view_occlusion_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    create_tabletop_operators,
    create_tabletop_skills,
)
from residual_controllers.tamp import PlanningComponents, run_tamp


def test_tabletop_view_occlusion_with_tamp():
    """Test TabletopViewOcclusionEnv pick-and-place with bilevel TAMP
    planner."""
    seed = 123

    base_env = TabletopViewOcclusionEnv(
        gui=False, num_objects=1, render_mode="rgb_array_external"
    )
    env = RecordVideo(base_env, "videos/planning")
    sim = TabletopViewOcclusionEnv(gui=False, num_objects=1)

    _, _ = env.reset(seed=seed)
    sim.reset(seed=seed)
    sim.set_state(env.get_state())

    gt_obs = sim.get_observation()

    types = TabletopTypes()
    predicates = TabletopPredicates(types)
    operators = create_tabletop_operators(types, predicates)

    abstractor = TabletopAbstractor(sim, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(gt_obs)
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
        initial_state=gt_obs,
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


def test_tabletop_object_occlusion_with_tamp():
    """Test TabletopObjectOcclusionEnv pick-and-pour with bilevel TAMP
    planner."""
    seed = 123

    base_env = TabletopObjectOcclusionEnv(gui=False, render_mode="rgb_array_external")
    env = RecordVideo(base_env, "videos/planning", name_prefix="pour_planning")
    sim = TabletopObjectOcclusionEnv(gui=False)

    _, _ = env.reset(seed=seed)
    sim.reset(seed=seed)
    sim.set_state(env.get_state())

    gt_obs = sim.get_observation()

    types = TabletopTypes()
    predicates = TabletopPourPredicates(types)
    operators = create_tabletop_pour_operators(types, predicates)

    abstractor = TabletopPourAbstractor(sim, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(gt_obs)
    print(f"Objects: {[o.name for o in objects]}")
    print(f"Initial atoms: {[str(a) for a in init_atoms]}")
    print(f"Goal atoms: {[str(a) for a in goal_atoms]}")

    components = PlanningComponents(
        types=types.as_set(),
        predicates=predicates,
        operators=operators,
        abstractor=abstractor,
    )

    skills = create_tabletop_pour_skills(types, operators, sim, abstractor)

    def transition_fn(_obs_state, action):
        next_obs, _, _, _, _ = sim.step(action)
        return next_obs

    goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

    plan, graph = run_tamp(
        components=components,
        skills=skills,
        initial_state=gt_obs,
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
    assert terminated, "Should reach goal (milk carton pouring above cup)"
    print("=" * 50)

    env.close()
    sim.close()


def test_nut_assembly_with_tamp():
    """Test NutAssemblyEnv pick-and-assemble with bilevel TAMP planner."""
    seed = 123

    base_env = NutAssemblyEnv(gui=False, render_mode="rgb_array_external")
    env = RecordVideo(base_env, "videos/planning", name_prefix="nut_assembly_planning")
    sim = NutAssemblyEnv(gui=False)

    _, _ = env.reset(seed=seed)
    sim.reset(seed=seed)
    sim.set_state(env.get_state())

    gt_obs = sim.get_observation()

    types = TabletopTypes()
    predicates = NutAssemblyPredicates(types)
    operators = create_nut_assembly_operators(types, predicates)

    abstractor = NutAssemblyAbstractor(sim, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(gt_obs)
    print(f"Objects: {[o.name for o in objects]}")
    print(f"Initial atoms: {[str(a) for a in init_atoms]}")
    print(f"Goal atoms: {[str(a) for a in goal_atoms]}")

    components = PlanningComponents(
        types=types.as_set(),
        predicates=predicates,
        operators=operators,
        abstractor=abstractor,
    )

    skills = create_nut_assembly_skills(types, operators, sim, abstractor)

    def transition_fn(_obs_state, action):
        next_obs, _, _, _, _ = sim.step(action)
        return next_obs

    goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

    plan, graph = run_tamp(
        components=components,
        skills=skills,
        initial_state=gt_obs,
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
    assert terminated, "Should reach goal (nut lowered onto peg)"
    print("=" * 50)

    env.close()
    sim.close()
