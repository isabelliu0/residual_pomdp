"""Tabletop TAMP system adapter for approaches."""

from __future__ import annotations

from typing import Any

import numpy as np

from residual_controllers.beliefs.structs import Belief
from residual_controllers.benchmarks.base_env import BaseTAMPSystem
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv
from residual_controllers.envs.tabletop_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    TabletopTypes,
    create_tabletop_operators,
    create_tabletop_skills,
)
from residual_controllers.information_gathering import NBVPlanner
from residual_controllers.information_gathering.nbv_planner import ViewpointCandidate
from residual_controllers.tamp.sesame_runner import run_tamp
from residual_controllers.tamp.structs import PlanningComponents


class TabletopPickTAMPSystem(BaseTAMPSystem[dict, np.ndarray]):
    """TabletopPickEnv wrapped as a TAMP system with information-gathering
    support.

    This system provides:
    - TAMP planning via SeSamE for pick operations
    - NBV-based information gathering for unknown/occluded objects
    - Belief state tracking via particle filter + occupancy grid
    """

    def __init__(
        self,
        seed: int | None = None,
        gui: bool = False,
        num_objects: int = 5,
        occlusion_prob: float = 0.7,
    ) -> None:
        self.gui = gui
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        self._nbv_planner: NBVPlanner | None = None
        self._target_object_id: int | None = None
        self._plan_env: TabletopPickEnv | None = None
        super().__init__(seed=seed)

    def _create_env(self) -> TabletopPickEnv:
        return TabletopPickEnv(
            gui=self.gui,
            num_objects=self.num_objects,
            occlusion_prob=self.occlusion_prob,
        )

    def _on_reset(self, _obs: dict, info: dict[str, Any]) -> None:
        assert self.env.robot is not None
        self._target_object_id = info.get("target_object_id")
        self._nbv_planner = NBVPlanner(
            robot=self.env.robot,
            camera_intrinsics=self.env.camera_intrinsics,
            camera_to_ee_transform=self.env.camera_to_ee_transform,
            n_viewpoint_samples=50,
            radius_range=(0.25, 0.35),
            elevation_range=(0.6, 1.3),
        )

    def plan(self, seed: int | None = None):
        """Run TAMP planning using SeSamE planner.

        Creates a separate planning env and syncs state from real env.
        """
        if self._plan_env is None:
            self._plan_env = TabletopPickEnv(
                gui=False,
                num_objects=self.num_objects,
                occlusion_prob=self.occlusion_prob,
            )
            self._plan_env.reset(seed=seed or self._seed)

        self._plan_env.set_state(self.env.get_state())
        obs = self._plan_env.get_observation()

        types = TabletopTypes()
        predicates = TabletopPredicates(types)
        operators = create_tabletop_operators(types, predicates)
        abstractor = TabletopAbstractor(self._plan_env, types, predicates)
        components = PlanningComponents(
            types=types.as_set(),
            predicates=predicates,
            operators=operators,
            abstractor=abstractor,
        )
        skills = create_tabletop_skills(types, operators, self._plan_env, abstractor)
        _, _, goal_atoms = abstractor.reset(obs)
        goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

        def transition_fn(_obs, action):
            next_obs, _, _, _, _ = self._plan_env.step(action)
            return next_obs

        plan, _ = run_tamp(
            components=components,
            skills=skills,
            initial_state=obs,
            goal=goal,
            state_abstractor=components.abstractor.step,
            transition_function=transition_fn,
            timeout=60.0,
            seed=seed or 0,
        )

        return plan

    def get_noop_action(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def get_belief(self) -> Belief | None:
        """Get current belief state from environment."""
        return self.env.belief

    def is_target_unknown(self, info: dict[str, Any]) -> bool:
        """Check if target object is in the unknown set."""
        belief = self.get_belief()
        if belief is None:
            return False
        target_id = info.get("target_object_id", self._target_object_id)
        if target_id is None:
            return False
        return target_id in belief.unknown_objects

    def select_viewpoint(
        self, target_id: int, seed: int | None = None
    ) -> ViewpointCandidate | None:
        """Select next-best-view for observing target object."""
        if self._nbv_planner is None:
            return None
        belief = self.get_belief()
        if belief is None:
            return None
        return self._nbv_planner.plan_for_object(
            belief=belief, object_id=target_id, z_range=(0.0, 0.1), seed=seed
        )

    def plan_to_viewpoint(
        self, viewpoint: ViewpointCandidate, n_waypoints: int = 20
    ) -> list[np.ndarray]:
        """Plan a trajectory to the viewpoint using linear interpolation.

        Returns list of joint configurations from current to target.
        """
        assert self.env.robot is not None
        current_joints = np.array(self.env.robot.get_joint_positions()[:7])
        target_joints = np.array(viewpoint.joint_config[:7])

        trajectory = []
        for i in range(n_waypoints + 1):
            alpha = i / n_waypoints
            waypoint = (1 - alpha) * current_joints + alpha * target_joints
            trajectory.append(waypoint)

        return trajectory

    def action_from_trajectory(
        self, current_waypoint: np.ndarray, next_waypoint: np.ndarray
    ) -> np.ndarray:
        """Compute action to move from current waypoint to next waypoint."""
        joint_delta = next_waypoint - current_waypoint
        action = np.concatenate([joint_delta, [0.0]]).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def is_viewpoint_reached(
        self, viewpoint: ViewpointCandidate, threshold: float = 0.05
    ) -> bool:
        """Check if robot joints are close to viewpoint configuration."""
        assert self.env.robot is not None
        current_joints = np.array(self.env.robot.get_joint_positions())
        target_joints = np.array(viewpoint.joint_config)
        return float(np.linalg.norm(target_joints[:7] - current_joints[:7])) < threshold

    def get_viewpoint_info_gain(self, viewpoint: ViewpointCandidate) -> float:
        """Get expected information gain for viewpoint."""
        return float(viewpoint.expected_ig)

    def close(self) -> None:
        """Clean up environments."""
        if self._plan_env is not None:
            self._plan_env.close()  # type: ignore[no-untyped-call]
            self._plan_env = None
        self.env.close()  # type: ignore[no-untyped-call]
