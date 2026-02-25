"""TAMP + NBV approach (inference only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from residual_controllers.approaches.base import (
    ApproachStepResult,
    BaseApproach,
)
from residual_controllers.benchmarks.base_env import BaseTAMPSystem


class ExecutionMode(Enum):
    """Execution mode for the approach."""

    PLANNING = auto()
    NBV = auto()
    EXECUTING = auto()
    DONE = auto()


@dataclass
class TampNbvMetrics:
    """Metrics for evaluating the TAMP + NBV approach."""

    nbv_calls: int = 0
    nbv_steps: int = 0
    nbv_viewpoints_selected: int = 0
    nbv_viewpoints_reached: int = 0
    nbv_early_termination: int = 0
    tamp_replans: int = 0
    plan_actions: int = 0
    total_steps: int = 0
    success: bool = False


@dataclass
class NBVState:
    """State for NBV execution."""

    target_id: int
    current_viewpoint: Any | None = None
    trajectory: list[np.ndarray] = field(default_factory=list)
    trajectory_idx: int = 0
    viewpoints_visited: int = 0
    last_info_gain: float = 0.0


@dataclass
class TampNbvConfig:
    """Configuration for the TAMP + NBV approach."""

    nbv_max_viewpoints: int = 10
    nbv_max_steps_per_viewpoint: int = 30
    nbv_min_info_gain: float = 10.0


class TampNbvApproach(BaseApproach[Any, Any]):
    """TAMP + closed-loop NBV information gathering."""

    def __init__(
        self,
        system: BaseTAMPSystem,
        seed: int,
        config: TampNbvConfig | None = None,
    ) -> None:
        super().__init__(system, seed)
        self.config = config or TampNbvConfig()
        self.metrics = TampNbvMetrics()
        self._mode = ExecutionMode.PLANNING
        self._plan_actions: list[Any] = []
        self._plan_idx = 0
        self._target_id: int | None = None
        self._nbv_state: NBVState | None = None
        self._last_info: dict[str, Any] = {}
        self._symbolic_plan: list[str] | None = None

    def reset(self, _obs: Any, info: dict[str, Any]) -> ApproachStepResult[Any]:
        self.metrics = TampNbvMetrics()
        self._mode = ExecutionMode.PLANNING
        self._plan_actions = []
        self._plan_idx = 0
        self._nbv_state = None
        self._last_info = info

        self._target_id = info.get("target_object_id")
        if self._target_id is None:
            self._mode = ExecutionMode.DONE
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={"error": "No target object specified"},
            )

        self._symbolic_plan = self.system.get_symbolic_plan()
        print(f"[TAMP] Initial symbolic plan: {self._symbolic_plan}")

        return self._decide_next_action(info)

    def step(
        self,
        _obs: Any,
        _reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[Any]:
        self._last_info = info
        self.metrics.total_steps += 1

        if terminated or truncated:
            self._mode = ExecutionMode.DONE
            self.metrics.success = terminated
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={"metrics": self.metrics},
            )

        return self._decide_next_action(info)

    def _decide_next_action(self, info: dict[str, Any]) -> ApproachStepResult[Any]:
        if self._should_do_nbv(info):
            return self._step_nbv(info)
        return self._step_plan()

    def _should_do_nbv(self, info: dict[str, Any]) -> bool:
        if hasattr(self.system, "is_target_unknown"):
            return self.system.is_target_unknown(info)
        return False

    def _step_nbv(self, _info: dict[str, Any]) -> ApproachStepResult[Any]:
        if self._nbv_state is None or self._mode != ExecutionMode.NBV:
            self._nbv_state = NBVState(target_id=self._target_id or 0)
            self._mode = ExecutionMode.NBV
            self.metrics.nbv_calls += 1
            print(f"[NBV] Entering NBV mode for target {self._target_id}")

        self.metrics.nbv_steps += 1

        if self._nbv_state.viewpoints_visited >= self.config.nbv_max_viewpoints:
            print(f"[NBV] Max viewpoints ({self.config.nbv_max_viewpoints}) reached")
            return self._exit_nbv_and_execute()

        trajectory_complete = (
            len(self._nbv_state.trajectory) == 0
            or self._nbv_state.trajectory_idx >= len(self._nbv_state.trajectory) - 1
        )

        if trajectory_complete and self._nbv_state.current_viewpoint is not None:
            self.metrics.nbv_viewpoints_reached += 1
            print("[NBV] Viewpoint reached, replanning...")

        if trajectory_complete:
            viewpoint = self._select_new_viewpoint()
            if viewpoint is None:
                print("[NBV] No valid viewpoint found, exiting NBV")
                self.metrics.nbv_early_termination += 1
                return self._exit_nbv_and_execute()

            if hasattr(self.system, "get_viewpoint_info_gain"):
                ig = self.system.get_viewpoint_info_gain(viewpoint)
                self._nbv_state.last_info_gain = ig
                if ig < self.config.nbv_min_info_gain:
                    print(f"[NBV] Info gain ({ig:.1f}) below threshold, exiting NBV")
                    self.metrics.nbv_early_termination += 1
                    return self._exit_nbv_and_execute()

            self._nbv_state.current_viewpoint = viewpoint
            self._nbv_state.viewpoints_visited += 1
            self.metrics.nbv_viewpoints_selected += 1

            assert hasattr(self.system, "plan_to_viewpoint") and hasattr(
                self.system, "action_from_trajectory"
            )
            self._nbv_state.trajectory = self.system.plan_to_viewpoint(viewpoint)
            self._nbv_state.trajectory_idx = 0
            assert (
                len(self._nbv_state.trajectory) >= 2
            ), "Trajectory must have at least 2 waypoints"
            print(
                f"[NBV] Selected viewpoint {self._nbv_state.viewpoints_visited}, "
                f"IG={self._nbv_state.last_info_gain:.1f}, "
                f"trajectory={len(self._nbv_state.trajectory)} waypoints"
            )

        assert (
            self._nbv_state.trajectory
        ), "Trajectory should not be empty at this point"
        curr_wp = self._nbv_state.trajectory[self._nbv_state.trajectory_idx]
        next_wp = self._nbv_state.trajectory[self._nbv_state.trajectory_idx + 1]
        action = self.system.action_from_trajectory(curr_wp, next_wp)  # type: ignore[attr-defined] # pylint: disable=line-too-long
        self._nbv_state.trajectory_idx += 1
        print(
            f"[NBV] Executing trajectory step {self._nbv_state.trajectory_idx}/{len(self._nbv_state.trajectory)}"  # pylint: disable=line-too-long
        )

        return ApproachStepResult(
            action=action,
            info={
                "mode": "nbv",
                "viewpoint_idx": self._nbv_state.viewpoints_visited,
                "trajectory_step": self._nbv_state.trajectory_idx,
                "info_gain": self._nbv_state.last_info_gain,
            },
        )

    def _select_new_viewpoint(self) -> Any | None:
        if not hasattr(self.system, "select_viewpoint"):
            return None
        if self._target_id is None:
            return None
        return self.system.select_viewpoint(self._target_id, seed=self._seed)

    def _exit_nbv_and_execute(self) -> ApproachStepResult[Any]:
        self._nbv_state = None
        self._replan()
        return self._step_plan()

    def _replan(self) -> None:
        print("[TAMP] Replanning...")
        self._plan_actions = []
        self._plan_idx = 0
        self.metrics.tamp_replans += 1
        plan = self.system.plan(seed=self._seed)
        if plan is not None:
            self._plan_actions = list(plan.actions)
            print(f"[TAMP] Plan has {len(self._plan_actions)} actions")
        else:
            print("[TAMP] Planning failed")

    def _step_plan(self) -> ApproachStepResult[Any]:
        self._mode = ExecutionMode.EXECUTING

        if not self._plan_actions:
            self._replan()

        if self._plan_idx >= len(self._plan_actions):
            self._mode = ExecutionMode.DONE
            self.metrics.success = False
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={"metrics": self.metrics, "mode": "done"},
            )

        action = self._plan_actions[self._plan_idx]
        self._plan_idx += 1
        self.metrics.plan_actions += 1

        return ApproachStepResult(
            action=action,
            info={"mode": "plan", "plan_step": self._plan_idx},
        )

    def get_metrics(self) -> TampNbvMetrics:
        """Get current metrics."""
        return self.metrics
