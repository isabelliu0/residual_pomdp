"""Base environment interface for approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from bilevel_planning.structs import Plan

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseTAMPSystem(Generic[ObsType, ActType], ABC):
    """Base class for TAMP systems used by approaches.

    This is the minimal interface for any TAMP system. It provides:
    - Environment access (reset, step via env)
    - TAMP planning (symbolic + motion planning)
    - Basic action utilities (noop, success checking)

    For information-gathering support, see InfoGatheringSystem protocol.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self.env = self._create_env()
        if seed is not None:
            self.env.reset(seed=seed)

    @abstractmethod
    def _create_env(self) -> Any:
        """Create and return the underlying environment."""

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset(seed=seed)
        self._on_reset(obs, info)
        return obs, info

    def _on_reset(self, _obs: ObsType, _info: dict[str, Any]) -> None:
        """Hook called after environment reset."""

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Step the environment."""
        return self.env.step(action)

    @abstractmethod
    def plan(self, seed: int | None = None) -> Plan | None:
        """Run TAMP planning and return a plan (or None if planning fails)."""

    @abstractmethod
    def get_noop_action(self) -> ActType:
        """Return a no-op action."""

    def get_symbolic_plan(self, _planner: str = "pyperplan") -> list[str] | None:
        """Generate a symbolic (PDDL-only) task plan."""
        return None

    def get_oig_ignored_objects(self) -> set[str]:
        """Return object names to exclude from the Object Interaction Graph.

        Override per environment to exclude always-known static objects
        (e.g. robot, fixed surfaces) that would otherwise merge all
        components.
        """
        return set()

    def get_subsequence_effects(
        self, _subsequence: list[str]
    ) -> tuple[set[Any], set[Any]]:
        """Compute net (add, delete) GroundAtom effects of a plan
        subsequence."""
        return set(), set()

    def is_object_set_unknown(self, _obj_names: set[str]) -> bool:
        """Return True if any object in obj_names is currently unknown in the
        belief."""
        return False

    def plan_for_goal(self, _goal_atoms: Any, seed: int | None = None) -> Plan | None:
        """Run bilevel planning toward the given goal atoms."""
        return self.plan(seed=seed)

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()


@runtime_checkable
class InfoGatheringSystem(Protocol):
    """Protocol for systems that support information-gathering.

    This protocol defines the interface for NBV-based information
    gathering. Systems implementing this can be used with
    TampNbvApproach.
    """

    def get_belief(self) -> Any | None:
        """Get current belief state."""

    def is_target_unknown(self, info: dict[str, Any]) -> bool:
        """Check if the target object is currently unknown/undetected."""

    def select_viewpoint(self, target_id: int, seed: int | None = None) -> Any | None:
        """Select next-best-view for information gathering.

        Returns a viewpoint object with joint_config attribute, or None
        if no valid viewpoint found.
        """

    def action_toward_viewpoint(self, viewpoint: Any) -> Any:
        """Compute action to move toward a viewpoint."""

    def is_viewpoint_reached(self, viewpoint: Any, threshold: float = 0.05) -> bool:
        """Check if robot has reached the target viewpoint."""

    def get_viewpoint_info_gain(self, viewpoint: Any) -> float:
        """Get expected information gain for a viewpoint."""
