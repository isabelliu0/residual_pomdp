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
