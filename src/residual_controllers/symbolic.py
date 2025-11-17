"""Minimal symbolic representation for planning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Predicate:
    """A predicate with name and argument types."""

    name: str
    arg_types: list[str] = field(default_factory=list)

    def pddl(self) -> str:
        """Convert to PDDL predicate declaration."""
        if not self.arg_types:
            return f"({self.name})"
        args = " ".join([f"?{chr(97+i)} - {t}" for i, t in enumerate(self.arg_types)])
        return f"({self.name} {args})"


@dataclass
class Atom:
    """A grounded or ungrounded atom."""

    pred_name: str
    args: list[str] = field(default_factory=list)

    def pddl(self) -> str:
        """Convert to PDDL atom."""
        if not self.args:
            return f"({self.pred_name})"
        return f"({self.pred_name} {' '.join(self.args)})"


@dataclass
class Not:
    """Negation."""

    expr: Atom

    def pddl(self) -> str:
        """Convert to PDDL negation."""
        return f"(not {self.expr.pddl()})"


@dataclass
class And:
    """Conjunction."""

    exprs: list[Atom | Not]

    def pddl(self) -> str:
        """Convert to PDDL conjunction."""
        if len(self.exprs) == 1:
            return self.exprs[0].pddl()
        inner = " ".join([e.pddl() for e in self.exprs])
        return f"(and {inner})"


@dataclass
class ActionSchema:
    """Schema for a PDDL action."""

    name: str
    parameters: list[tuple[str, str]] = field(
        default_factory=list
    )  # [(var_name, type)]
    preconditions: And | Atom | Not | None = None
    effects: And | list[Atom | Not] = field(default_factory=list)

    def pddl(self) -> str:
        """Convert to PDDL action."""
        lines = [f"  (:action {self.name}"]
        if self.parameters:
            params_str = " ".join([f"{var} - {typ}" for var, typ in self.parameters])
            lines.append(f"    :parameters ({params_str})")
        if self.preconditions:
            if isinstance(self.preconditions, (Atom, Not)):
                lines.append(f"    :precondition {self.preconditions.pddl()}")
            else:
                lines.append(f"    :precondition {self.preconditions.pddl()}")
        if self.effects:
            if isinstance(self.effects, list):
                effects = And(self.effects)
            else:
                effects = self.effects
            lines.append(f"    :effect {effects.pddl()}")
        lines.append("  )")
        return "\n".join(lines)


@dataclass
class ProblemSpec:
    """Specification for a planning problem."""

    goal: And | Atom
    predicates: list[Predicate] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    action_schemas: list[ActionSchema] = field(default_factory=list)
    objects: dict[str, list[str]] = field(default_factory=dict)  # type -> [obj_names]
    init: list[Atom] = field(default_factory=list)

    def to_pddl_domain(self) -> str:
        """Generate PDDL domain."""
        lines = ["(define (domain cover2d)"]
        if self.types:
            types_str = " ".join(self.types)
            lines.append(f"  (:types {types_str})")
        lines.append("  (:predicates")
        for pred in self.predicates:
            lines.append(f"    {pred.pddl()}")
        lines.append("  )")
        for action in self.action_schemas:
            lines.append("")
            lines.append(action.pddl())
        lines.append(")")
        return "\n".join(lines)

    def to_pddl_problem(self) -> str:
        """Generate PDDL problem."""
        lines = ["(define (problem cover2d-problem)", "  (:domain cover2d)"]
        if self.objects:
            objs_by_type = []
            for typ, objs in self.objects.items():
                objs_str = " ".join(objs)
                objs_by_type.append(f"{objs_str} - {typ}")
            lines.append(f"  (:objects {' '.join(objs_by_type)})")
        lines.append("  (:init")
        for atom in self.init:
            lines.append(f"    {atom.pddl()}")
        lines.append("  )")
        if self.goal:
            lines.append(f"  (:goal {self.goal.pddl()})")
        lines.append(")")
        return "\n".join(lines)
