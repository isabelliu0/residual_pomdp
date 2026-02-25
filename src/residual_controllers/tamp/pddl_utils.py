"""Utilities for generating PDDL strings and running symbolic planners."""

from __future__ import annotations

from prpl_utils.pddl_planning import run_pddl_planner
from relational_structs import GroundAtom, LiftedOperator, Object, Predicate, Type


def _predicate_to_pddl(pred: Predicate) -> str:
    args = " ".join(f"?v{i} - {t.name}" for i, t in enumerate(pred.types))
    return f"({pred.name} {args})" if args else f"({pred.name})"


def _lifted_atom_to_pddl(atom) -> str:
    args = " ".join(v.name for v in atom.entities)
    return f"({atom.predicate.name} {args})" if args else f"({atom.predicate.name})"


def _ground_atom_to_pddl(atom: GroundAtom) -> str:
    args = " ".join(o.name for o in atom.entities)
    return f"({atom.predicate.name} {args})" if args else f"({atom.predicate.name})"


def _conjunction(strs: list[str]) -> str:
    if not strs:
        return "()"
    if len(strs) == 1:
        return strs[0]
    body = " ".join(strs)
    return f"(and {body})"


def _operator_to_pddl_action(op: LiftedOperator) -> str:
    params = " ".join(f"{v.name} - {v.type.name}" for v in op.parameters)
    pre = _conjunction(sorted(_lifted_atom_to_pddl(a) for a in op.preconditions))
    adds = sorted(_lifted_atom_to_pddl(a) for a in op.add_effects)
    dels = sorted(f"(not {_lifted_atom_to_pddl(a)})" for a in op.delete_effects)
    eff = _conjunction(adds + dels)
    return (
        f"  (:action {op.name}\n"
        f"    :parameters ({params})\n"
        f"    :precondition {pre}\n"
        f"    :effect {eff}\n"
        f"  )"
    )


def generate_pddl_domain(
    domain_name: str,
    types: set[Type],
    predicates: set[Predicate],
    operators: set[LiftedOperator],
) -> str:
    """Generate a PDDL domain string from the given components."""
    types_str = " ".join(sorted(t.name for t in types))
    preds_str = "\n    ".join(sorted(_predicate_to_pddl(p) for p in predicates))
    actions_str = "\n\n".join(
        _operator_to_pddl_action(op) for op in sorted(operators, key=lambda o: o.name)
    )
    return (
        f"(define (domain {domain_name})\n"
        f"  (:requirements :strips :typing)\n"
        f"  (:types {types_str})\n"
        f"  (:predicates\n"
        f"    {preds_str}\n"
        f"  )\n\n"
        f"{actions_str}\n"
        f")"
    )


def generate_pddl_problem(
    problem_name: str,
    domain_name: str,
    objects: set[Object],
    init_atoms: set[GroundAtom],
    goal_atoms: set[GroundAtom],
) -> str:
    """Generate a PDDL problem string from the given components."""
    by_type: dict[str, list[str]] = {}
    for obj in objects:
        by_type.setdefault(obj.type.name, []).append(obj.name)
    obj_lines = "\n".join(
        f"    {' '.join(sorted(names))} - {tname}"
        for tname, names in sorted(by_type.items())
    )
    init_str = "\n    ".join(sorted(_ground_atom_to_pddl(a) for a in init_atoms))
    goal_strs = sorted(_ground_atom_to_pddl(a) for a in goal_atoms)
    goal_str = _conjunction(goal_strs)
    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain {domain_name})\n"
        f"  (:objects\n"
        f"{obj_lines}\n"
        f"  )\n"
        f"  (:init\n"
        f"    {init_str}\n"
        f"  )\n"
        f"  (:goal {goal_str})\n"
        f")"
    )


def build_object_interaction_graph(
    symbolic_plan: list[str],
    ignored_objects: set[str] | None = None,
) -> list[tuple[list[str], set[str]]]:
    """Partition a symbolic plan into subsequences via an Object Interaction
    Graph.

    Each action's argument objects are nodes; objects co-occurring in
    the same action are connected by an edge. Connected components
    define object sets; consecutive plan actions whose objects share the
    same component are grouped into one subsequence.

    Returns a list of (action_subsequence, object_set) pairs.
    """
    ignored = ignored_objects or set()

    parsed: list[tuple[str, list[str]]] = []
    for action_str in symbolic_plan:
        tokens = action_str.strip("() ").split()
        args = [t for t in tokens[1:] if t not in ignored]
        parsed.append((action_str, args))

    all_objects: set[str] = set()
    for _, args in parsed:
        all_objects.update(args)

    parent = {o: o for o in all_objects}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    for _, args in parsed:
        for i in range(1, len(args)):
            _union(args[0], args[i])

    if not parsed:
        return []

    subsequences: list[tuple[list[str], set[str]]] = []
    current_actions: list[str] = []
    current_objects: set[str] = set()
    current_key: str | None = None

    for action_str, args in parsed:
        key = _find(args[0]) if args else None
        if key != current_key and current_actions:
            subsequences.append((current_actions, current_objects))
            current_actions = []
            current_objects = set()
        current_key = key
        current_actions.append(action_str)
        current_objects.update(args)

    if current_actions:
        subsequences.append((current_actions, current_objects))

    return subsequences


def run_symbolic_planner(
    domain_name: str,
    types: set[Type],
    predicates: set[Predicate],
    operators: set[LiftedOperator],
    objects: set[Object],
    init_atoms: set[GroundAtom],
    goal_atoms: set[GroundAtom],
    planner: str = "pyperplan",
) -> list[str] | None:
    """Generate PDDL domain + problem and run a pure symbolic planner."""
    domain_str = generate_pddl_domain(domain_name, types, predicates, operators)
    problem_str = generate_pddl_problem(
        f"{domain_name}-problem", domain_name, objects, init_atoms, goal_atoms
    )
    return run_pddl_planner(domain_str, problem_str, planner=planner)
