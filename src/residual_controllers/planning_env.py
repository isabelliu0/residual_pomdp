"""Minimal planning environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PlanningProblemSpec:
    """Specification for a planning problem."""

    domain_pddl: str
    problem_pddl: str


class PlanningEnv(ABC):
    """Minimal interface for environments that support task planning."""

    @abstractmethod
    def get_planning_problem(self) -> PlanningProblemSpec:
        """Generate PDDL domain and problem files for the current environment
        state.

        Returns:
            PlanningProblemSpec with domain_pddl and problem_pddl strings
        """
        raise NotImplementedError
