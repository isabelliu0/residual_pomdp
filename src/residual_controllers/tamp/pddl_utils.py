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
