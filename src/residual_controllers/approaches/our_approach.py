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
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph


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

    target_label: str
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
        self._target_label: str | None = None
        self._nbv_state: NBVState | None = None
        self._last_info: dict[str, Any] = {}
        self._symbolic_plan: list[str] | None = None
        self._oig_subsequences: list[tuple[list[str], set[str]]] = []
        self._current_subseq_idx: int = 0

    def reset(self, _obs: Any, info: dict[str, Any]) -> ApproachStepResult[Any]:
        self.metrics = TampNbvMetrics()
        self._mode = ExecutionMode.PLANNING
        self._plan_actions = []
        self._plan_idx = 0
        self._nbv_state = None
        self._last_info = info
        self._oig_subsequences = []
        self._current_subseq_idx = 0

        self._target_label = info.get("target_object_label")
        if self._target_label is None:
            self._mode = ExecutionMode.DONE
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={"error": "No target object specified"},
            )

        self._symbolic_plan = self.system.get_symbolic_plan()
        print(f"[TAMP] Initial symbolic plan: {self._symbolic_plan}")

        if self._symbolic_plan is not None:
            ignored = self.system.get_oig_ignored_objects()
            self._oig_subsequences = build_object_interaction_graph(
                self._symbolic_plan, ignored
            )
            print(f"[TAMP] OIG subsequences: {self._oig_subsequences}")

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
        return self._step_oig_plan()

    def _should_do_nbv(self, _info: dict[str, Any]) -> bool:
        if self._current_subseq_idx < len(self._oig_subsequences):
            _, obj_set = self._oig_subsequences[self._current_subseq_idx]
            return self.system.is_object_set_unknown(obj_set)
        return False

    def _step_nbv(self, _info: dict[str, Any]) -> ApproachStepResult[Any]:
        if self._nbv_state is None or self._mode != ExecutionMode.NBV:
            self._nbv_state = NBVState(target_label=self._target_label or "")
            self._mode = ExecutionMode.NBV
            self.metrics.nbv_calls += 1
            print(f"[NBV] Entering NBV mode for target {self._target_label}")

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
        if self._target_label is None:
            return None
        return self.system.select_viewpoint(self._target_label, seed=self._seed)

    def _exit_nbv_and_execute(self) -> ApproachStepResult[Any]:
        self._nbv_state = None
        self._plan_actions = []
        self._plan_idx = 0
        return self._decide_next_action(self._last_info)

    def _step_oig_plan(self) -> ApproachStepResult[Any]:
        if self._current_subseq_idx >= len(self._oig_subsequences):
            self._mode = ExecutionMode.DONE
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={"metrics": self.metrics, "mode": "done"},
            )

        if not self._plan_actions:
            self._replan_for_current_subsequence()

        if not self._plan_actions:
            self._mode = ExecutionMode.DONE
            return ApproachStepResult(
                action=self.system.get_noop_action(),
                terminate=True,
                info={
                    "metrics": self.metrics,
                    "mode": "done",
                    "error": "planning_failed",
                },
            )

        if self._plan_idx >= len(self._plan_actions):
            print(f"[TAMP] Subsequence {self._current_subseq_idx} complete, advancing")
            self._current_subseq_idx += 1
            self._plan_actions = []
            self._plan_idx = 0
            return self._step_oig_plan()

        self._mode = ExecutionMode.EXECUTING
        action = self._plan_actions[self._plan_idx]
        self._plan_idx += 1
        self.metrics.plan_actions += 1
        return ApproachStepResult(
            action=action,
            info={
                "mode": "plan",
                "plan_step": self._plan_idx,
                "subseq_idx": self._current_subseq_idx,
            },
        )

    def _replan_for_current_subsequence(self) -> None:
        self._plan_actions = []
        self._plan_idx = 0
        self.metrics.tamp_replans += 1

        subseq_actions, _ = self._oig_subsequences[self._current_subseq_idx]
        net_add, _ = self.system.get_subsequence_effects(subseq_actions)

        print(
            f"[TAMP] Replanning for subsequence {self._current_subseq_idx} "
            f"with goal: {net_add}"
        )

        plan = (
            self.system.plan_for_goal(net_add, seed=self._seed)
            if net_add
            else self.system.plan(seed=self._seed)
        )

        if plan is not None:
            self._plan_actions = list(plan.actions)
            print(f"[TAMP] Plan has {len(self._plan_actions)} actions")
        else:
            print("[TAMP] Planning failed")

    def get_metrics(self) -> TampNbvMetrics:
        """Get current metrics."""
        return self.metrics
