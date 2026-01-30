"""Belief abstraction interface for TAMP integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from bilevel_planning.structs import RelationalAbstractGoal, RelationalAbstractState
from relational_structs import GroundAtom, LiftedOperator, Object, Predicate, Type

_O = TypeVar("_O")


@dataclass
class TypeContainer(ABC):
    """Abstract base class for type containers."""

    @abstractmethod
    def as_set(self) -> set[Type]:
        """Convert to set of types."""


@dataclass
class PredicateContainer(ABC):
    """Abstract base class for predicate containers."""

    @abstractmethod
    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""

    @abstractmethod
    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""


class BeliefAbstractor(ABC, Generic[_O]):
    """Abstract interface for converting belief states to relational abstract
    states."""

    @abstractmethod
    def reset(self, obs: _O) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset abstractor with observation, return objects, init atoms, and
        goal atoms."""

    @abstractmethod
    def step(self, obs: _O) -> RelationalAbstractState:
        """Step abstractor with observation, return abstract state."""

    def create_abstract_goal(
        self, goal_atoms: set[GroundAtom], state_abstractor
    ) -> RelationalAbstractGoal:
        """Helper to create a RelationalAbstractGoal from goal atoms."""
        return RelationalAbstractGoal(
            atoms=goal_atoms, state_abstractor=state_abstractor
        )


@dataclass
class PlanningComponents(Generic[_O]):
    """Container for all planning components needed for TAMP."""

    types: set[Type]
    predicates: PredicateContainer
    operators: set[LiftedOperator]
    abstractor: BeliefAbstractor[_O]


def observation_to_abstract_state(
    atoms: set[GroundAtom], objects: set[Object]
) -> RelationalAbstractState:
    """Helper to create RelationalAbstractState from atoms and objects."""
    return RelationalAbstractState(atoms=atoms, objects=objects)
