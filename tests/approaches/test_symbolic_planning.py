"""Tests for optimistic belief abstraction and PDDL symbolic planning."""

from residual_controllers.beliefs import get_mean_state
from residual_controllers.benchmarks import TabletopPickTAMPSystem
from residual_controllers.envs.tabletop_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    TabletopTypes,
    create_tabletop_operators,
)
from residual_controllers.tamp.pddl_utils import (
    build_object_interaction_graph,
    generate_pddl_domain,
    generate_pddl_problem,
)


def test_optimistic_atoms_and_symbolic_plan():
    """Print optimistic init atoms and PDDL symbolic plan."""
    system = TabletopPickTAMPSystem(seed=42, gui=False, num_objects=5)
    _, _ = system.reset(seed=42)

    assert system._plan_env is not None  # pylint: disable=protected-access
    assert system.env.belief is not None

    held_obs = system._plan_env.get_obs_from_mean(  # pylint: disable=protected-access
        get_mean_state(system.env.belief), system.env.scene.object_ids
    )

    types = TabletopTypes()
    predicates = TabletopPredicates(types)
    operators = create_tabletop_operators(types, predicates)
    abstractor = TabletopAbstractor(
        system._plan_env, types, predicates  # pylint: disable=protected-access
    )
    objects, mean_init_atoms, goal_atoms = abstractor.reset(held_obs)

    optimistic_atoms = abstractor.get_atoms_from_belief_particles(
        system.env.belief.particles, system.env.scene.object_ids, held_obs
    )

    print("\n=== Objects ===")
    print([o.name for o in sorted(objects, key=lambda o: o.name)])

    print("\n=== Mean init atoms ===")
    for a in sorted(str(a) for a in mean_init_atoms):
        print(f"  {a}")

    print("\n=== Optimistic init atoms (union across all particles) ===")
    for a in sorted(str(a) for a in optimistic_atoms):
        print(f"  {a}")

    extra = optimistic_atoms - mean_init_atoms
    print(f"\n=== Atoms only in optimistic (not in mean): {len(extra)} ===")
    for a in sorted(str(a) for a in extra):
        print(f"  {a}")

    print("\n=== PDDL domain ===")
    print(
        generate_pddl_domain("tabletop", types.as_set(), predicates.as_set(), operators)
    )

    print("\n=== PDDL problem ===")
    print(
        generate_pddl_problem(
            "tabletop-problem", "tabletop", objects, optimistic_atoms, goal_atoms
        )
    )

    symbolic_plan = system.get_symbolic_plan()
    print("\n=== Symbolic plan ===")
    print(symbolic_plan)

    assert symbolic_plan is not None
    assert len(symbolic_plan) >= 2  # at least pick + place

    system.close()


def test_oig_single_subsequence():
    """OIG on the tabletop plan produces one subsequence with the target
    object."""
    system = TabletopPickTAMPSystem(seed=42, gui=False, num_objects=5)
    system.reset(seed=42)

    symbolic_plan = system.get_symbolic_plan()
    assert symbolic_plan is not None

    ignored = system.get_oig_ignored_objects()
    subsequences = build_object_interaction_graph(symbolic_plan, ignored)

    print("\n=== OIG subsequences ===")
    for actions, obj_set in subsequences:
        print(f"  actions={actions}, objects={obj_set}")

    assert len(subsequences) == 1
    _, obj_set = subsequences[0]
    assert "a" in obj_set
    assert "robot" not in obj_set
    assert "table" not in obj_set
    assert "target_area" not in obj_set

    system.close()


def test_nbv_triggered():
    """With full occlusion, NBV should be triggered for the target's
    subsequence, and net goal effects should be non-empty."""
    system = TabletopPickTAMPSystem(
        seed=42, gui=False, num_objects=5, occlusion_prob=0.7
    )
    system.reset(seed=42)

    belief = system.env.belief
    assert belief is not None

    symbolic_plan = system.get_symbolic_plan()
    assert (
        symbolic_plan is not None
    ), "Expected a symbolic plan even under full occlusion"
    print(f"\n=== Symbolic plan ===\n{symbolic_plan}")

    ignored = system.get_oig_ignored_objects()
    subsequences = build_object_interaction_graph(symbolic_plan, ignored)
    assert len(subsequences) >= 1

    subseq_actions, obj_set = subsequences[0]
    print("\n=== First subsequence ===")
    print(f"  actions: {subseq_actions}")
    print(f"  object set: {obj_set}")

    nbv_needed = system.is_object_set_unknown(obj_set)
    print(f"NBV needed for first subsequence: {nbv_needed}")
    assert nbv_needed, "Expected target object to be unknown"

    net_add, net_del = system.get_subsequence_effects(subseq_actions)
    print("\n=== Net add effects ===")
    for a in sorted(str(a) for a in net_add):
        print(f"  {a}")
    print("\n=== Net delete effects ===")
    for a in sorted(str(a) for a in net_del):
        print(f"  {a}")

    assert len(net_add) > 0, "Expected non-empty add effects from pick+place"

    system.close()
