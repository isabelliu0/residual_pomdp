"""Tabletop TAMP system adapter for approaches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose

from residual_controllers.beliefs import get_best_particle_state
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
from residual_controllers.tamp.pddl_utils import (
    compute_subsequence_effects,
    run_symbolic_planner,
)
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
        num_objects: int = 1,
        occlusion_prob: float = 0.7,
    ) -> None:
        self.gui = gui
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        self._nbv_planner: NBVPlanner | None = None
        self._target_object_label: str | None = None
        self._plan_env: TabletopPickEnv | None = None
        self.abstractor: TabletopAbstractor | None = None
        super().__init__(seed=seed)

    def _create_env(self) -> TabletopPickEnv:
        return TabletopPickEnv(
            gui=self.gui,
            num_objects=self.num_objects,
            occlusion_prob=self.occlusion_prob,
        )

    def _on_reset(self, _obs: dict, info: dict[str, Any]) -> None:
        assert self.env.robot is not None
        self._target_object_label = info.get("target_object_label")
        self._nbv_planner = NBVPlanner(
            robot=self.env.robot,
            camera_intrinsics=self.env.camera_intrinsics,
            camera_to_ee_transform=self.env.camera_to_ee_transform,
            n_viewpoint_samples=50,
            radius_range=(0.25, 0.35),
            elevation_range=(0.6, 1.3),
        )
        if self._plan_env is None:
            self._plan_env = TabletopPickEnv(
                gui=False,
                num_objects=self.num_objects,
                occlusion_prob=self.occlusion_prob,
            )
            self._plan_env.reset(seed=self._seed)

    def plan(self, seed: int | None = None):
        """Run TAMP planning using SeSamE planner."""
        return self._run_plan(goal_atoms_override=None, seed=seed)

    def plan_for_goal(self, goal_atoms: Any, seed: int | None = None):
        """Run bilevel planning toward specific goal atoms."""
        return self._run_plan(goal_atoms_override=goal_atoms, seed=seed)

    def _run_plan(self, goal_atoms_override: Any, seed: int | None):
        assert self._plan_env is not None

        obs = self._build_plan_obs_from_belief(self.env.belief)

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
        _, _, default_goal_atoms = abstractor.reset(obs)
        goal_atoms = (
            goal_atoms_override
            if goal_atoms_override is not None
            else default_goal_atoms
        )
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

    def get_symbolic_plan(self, planner: str = "pyperplan") -> list[str] | None:
        assert self._plan_env is not None

        held_obs = self._build_plan_obs_from_belief(self.env.belief)

        types = TabletopTypes()
        predicates = TabletopPredicates(types)
        operators = create_tabletop_operators(types, predicates)
        abstractor = TabletopAbstractor(self._plan_env, types, predicates)
        objects, mean_init_atoms, goal_atoms = abstractor.reset(held_obs)
        self.abstractor = abstractor

        if self.env.belief is not None:
            init_atoms = abstractor.get_atoms_from_belief_particles(
                self.env.belief.particles, held_obs
            )
        else:
            init_atoms = mean_init_atoms

        return run_symbolic_planner(
            domain_name="tabletop",
            types=types.as_set(),
            predicates=predicates.as_set(),
            operators=operators,
            objects=objects,
            init_atoms=init_atoms,
            goal_atoms=goal_atoms,
            planner=planner,
        )

    def _build_plan_obs_from_belief(self, belief: Belief | None) -> dict:
        """Set up plan_env for planning without leaking GT object poses.

        Robot joints and target area come from the actual env (known
        exactly). Object poses come from the best particle (highest-
        weight); if no belief, syncs object poses from the actual env
        directly.
        """
        assert self._plan_env is not None
        assert self._plan_env.scene is not None
        assert self.env.scene is not None

        self._plan_env.robot.set_joints(list(self.env.robot.get_joint_positions()))

        target_area_pose = get_pose(
            self.env.scene.target_area_id, self.env.physics_client_id
        )
        set_pose(
            self._plan_env.scene.target_area_id,
            target_area_pose,
            self._plan_env.physics_client_id,
        )

        if belief is not None:
            best = get_best_particle_state(belief)
            for label, plan_obj_id in self._plan_env.scene.label_to_id.items():
                pose = best.object_poses.get(label)
                if pose is not None:
                    set_pose(
                        plan_obj_id,
                        Pose(position=pose[:3], orientation=pose[3:]),
                        self._plan_env.physics_client_id,
                    )
        else:
            for label in self._plan_env.scene.label_to_id:
                main_id = self.env.scene.label_to_id[label]
                plan_id = self._plan_env.scene.label_to_id[label]
                set_pose(
                    plan_id,
                    get_pose(main_id, self.env.physics_client_id),
                    self._plan_env.physics_client_id,
                )

        return self._plan_env.get_observation()

    def get_noop_action(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def get_oig_ignored_objects(self) -> set[str]:
        return {"robot", "table", "target_area"}

    def get_subsequence_effects(
        self, subsequence: list[str]
    ) -> tuple[set[Any], set[Any]]:
        if self.abstractor is None:
            return set(), set()
        types = TabletopTypes()
        predicates = TabletopPredicates(types)
        operators = create_tabletop_operators(types, predicates)
        objects_by_name = {
            obj.name.lower(): obj
            for obj in self.abstractor._pybullet_ids  # pylint: disable=protected-access
        }
        return compute_subsequence_effects(subsequence, operators, objects_by_name)

    def _name_to_main_id_map(self) -> dict[str, int]:
        """Map symbolic object names (lowercase) to main-env body IDs."""
        if self.abstractor is None or self._plan_env is None:
            return {}
        assert self.env.scene is not None
        return {
            obj.name.lower(): self.env.scene.label_to_id[obj.name.upper()]
            for obj in self.abstractor._pybullet_ids  # pylint: disable=protected-access
            if obj.name.upper() in self.env.scene.label_to_id
        }

    def is_object_set_unknown(self, obj_names: set[str]) -> bool:
        belief = self.get_belief()
        if belief is None:
            return False
        label_map = self.get_object_labels_for_names(obj_names)
        return any(label in belief.unknown_objects for label in label_map.values())

    def get_object_labels_for_names(self, names: set[str]) -> dict[str, str]:
        """Map symbolic names to object labels (e.g. 'a' -> 'A')."""
        if self.abstractor is None:
            return {}
        scene_labels = self.env.scene.label_to_id if self.env.scene else {}
        pddl_to_label = {
            obj.name.lower(): obj.name.upper()
            for obj in self.abstractor._pybullet_ids  # pylint: disable=protected-access
            if obj.name.upper() in scene_labels
        }
        return {name: pddl_to_label[name] for name in names if name in pddl_to_label}

    def get_ground_truth_object_poses(self, obj_ids: list[int]) -> dict[int, tuple]:
        """Get ground-truth object poses directly from PyBullet (for training
        only)."""
        poses = {}
        for obj_id in obj_ids:
            pos, quat = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self.env.physics_client_id
            )
            poses[obj_id] = pos + quat
        return poses

    def plan_for_goal_with_belief(
        self, goal_atoms: Any, belief_override: Any, seed: int | None = None
    ):
        """Plan toward goal atoms using a specified belief (for training data
        collection)."""
        original_belief = self.env.belief
        self.env.belief = belief_override
        try:
            return self._run_plan(goal_atoms_override=goal_atoms, seed=seed)
        finally:
            self.env.belief = original_belief

    def get_belief(self) -> Belief | None:
        """Get current belief state from environment."""
        return self.env.belief

    def is_target_unknown(self, info: dict[str, Any]) -> bool:
        """Check if target object is in the unknown set."""
        belief = self.get_belief()
        if belief is None:
            return False
        target_label = info.get("target_object_label", self._target_object_label)
        if target_label is None:
            return False
        return target_label in belief.unknown_objects

    def select_viewpoint(
        self, target_label: str, seed: int | None = None
    ) -> ViewpointCandidate | None:
        """Select next-best-view for observing target object."""
        if self._nbv_planner is None:
            return None
        belief = self.get_belief()
        if belief is None:
            return None
        return self._nbv_planner.plan_for_object(
            belief=belief, object_label=target_label, z_range=(0.0, 0.1), seed=seed
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
