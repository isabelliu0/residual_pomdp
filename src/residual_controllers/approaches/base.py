"""Base class for all approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from residual_controllers.benchmarks.base_env import BaseTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class ApproachStepResult(Generic[ActType]):
    """Result from an approach step."""

    action: ActType
    terminate: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseApproach(Generic[ObsType, ActType], ABC):
    """Base class for all approaches."""

    def __init__(self, system: BaseTAMPSystem[ObsType, ActType], seed: int) -> None:
        """Initialize the approach with the given TAMP system and random
        seed."""
        self.system = system
        self._seed = seed

    @abstractmethod
    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset the approach."""

    @abstractmethod
    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Take a step in the approach."""
