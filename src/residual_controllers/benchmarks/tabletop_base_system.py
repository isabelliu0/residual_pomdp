"""Base TAMP system for tabletop environments."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import pybullet as p
from bilevel_planning.structs import LiftedSkill, Plan
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose

from residual_controllers.beliefs import get_best_particle_state
from residual_controllers.beliefs.structs import Belief
from residual_controllers.benchmarks.base_env import BaseTAMPSystem
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_tamp_base import TabletopBaseAbstractor
from residual_controllers.geometry import invert_pose
from residual_controllers.information_gathering import NBVPlanner
from residual_controllers.information_gathering.nbv_planner import (
    ViewpointCandidate,
    compute_object_target_center,
)
from residual_controllers.tamp.collision_utils import (
    run_smooth_motion_planning_to_pose_with_surface_check,
)
from residual_controllers.tamp.pddl_utils import (
    compute_subsequence_effects,
    run_symbolic_planner,
)
from residual_controllers.tamp.sesame_runner import run_tamp
from residual_controllers.tamp.structs import PlanningComponents


class TabletopBaseSystem(BaseTAMPSystem[dict, np.ndarray]):
    """Common TAMP system infrastructure for tabletop environments."""

    env: TabletopBaseEnv

    def __init__(self, seed: int | None = None, gui: bool = False) -> None:
        self.gui = gui
        self._plan_env: TabletopBaseEnv | None = None
        self._nbv_planner: NBVPlanner | None = None
        self._target_object_label: str | None = None
        self.abstractor: TabletopBaseAbstractor | None = None
        self._last_plan_components: tuple | None = None
        self._nbv_debug_ids: list[int] = []
        super().__init__(seed=seed)

    @abstractmethod
    def _create_plan_env(self) -> TabletopBaseEnv:
        """Create a headless duplicate of the env used for planning."""

    @abstractmethod
    def _get_planning_components(
        self,
    ) -> tuple[PlanningComponents, set[LiftedSkill]]:
        """Return TAMP planning components for this environment."""

    def _on_reset(self, _obs: dict, info: dict[str, Any]) -> None:
        self._target_object_label = info.get("target_object_label")
        if self._plan_env is None:
            self._plan_env = self._create_plan_env()
            self._plan_env.reset(seed=self._seed)
        assert self._plan_env is not None
        assert self._plan_env.scene is not None
        self._nbv_planner = NBVPlanner(
            robot=self._plan_env.robot,
            camera_intrinsics=self.env.camera_intrinsics,
            camera_to_ee_transform=self.env.camera_to_ee_transform,
            n_viewpoint_samples=50,
            radius_range=(0.25, 0.35),
            elevation_range=(0.6, 1.3),
            object_ids=set(self._plan_env.scene.object_ids),
            physics_client_id=self._plan_env.physics_client_id,
        )

    def get_noop_action(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def plan(self, seed: int | None = None) -> Plan | None:
        return self._run_plan(goal_atoms_override=None, seed=seed)

    def plan_for_goal(self, goal_atoms: Any, seed: int | None = None) -> Plan | None:
        return self._run_plan(goal_atoms_override=goal_atoms, seed=seed)

    def _run_plan(self, goal_atoms_override: Any, seed: int | None) -> Plan | None:
        assert self._plan_env is not None

        obs = self._build_plan_obs_from_belief(self.env.belief)

        components, skills = self._get_planning_components()
        abstractor = components.abstractor
        _, _, default_goal_atoms = abstractor.reset(obs)
        self._last_plan_components = (components, skills)
        goal_atoms = (
            goal_atoms_override
            if goal_atoms_override is not None
            else default_goal_atoms
        )
        goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

        obs_poses = obs.get("object_poses")
        obs_joints = list(obs.get("joint_positions", []))
        obs_held_idx = int(obs.get("held_object_idx", np.array([-1]))[0])

        def transition_fn(_obs: dict, action: np.ndarray) -> dict:
            """Transition function for planning."""
            if _obs is obs:
                # New trajectory attempt: restore plan_env to initial state so
                # each sampling attempt starts from the same configuration.
                assert self._plan_env is not None
                assert self._plan_env.scene is not None
                if obs_poses is not None:
                    for i, obj_id in enumerate(self._plan_env.scene.object_ids):
                        if i < len(obs_poses):
                            set_pose(
                                obj_id,
                                Pose(
                                    position=tuple(obs_poses[i, :3]),
                                    orientation=tuple(obs_poses[i, 3:]),
                                ),
                                self._plan_env.physics_client_id,
                            )
                if obs_joints:
                    self._plan_env.robot.set_joints(obs_joints)
                if obs_held_idx >= 0:
                    self._plan_env.held_object_id = self._plan_env.scene.object_ids[
                        obs_held_idx
                    ]
                else:
                    self._plan_env.held_object_id = None
                    self._plan_env.grasp_transform = None
            next_obs, _, _, _, _ = self._plan_env.step(action)  # type: ignore[union-attr]  # pylint: disable=line-too-long
            return next_obs

        plan, _ = run_tamp(
            components=components,
            skills=skills,
            initial_state=obs,
            goal=goal,
            state_abstractor=components.abstractor.step,
            transition_function=transition_fn,
            timeout=90.0,
            seed=seed or 0,
        )
        return plan

    def get_symbolic_plan(self, planner: str = "pyperplan") -> list[str] | None:
        assert self._plan_env is not None

        held_obs = self._build_plan_obs_from_belief(self.env.belief)

        components, _ = self._get_planning_components()
        assert isinstance(components.abstractor, TabletopBaseAbstractor)
        abstractor = components.abstractor
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
            types=components.types,
            predicates=components.predicates.as_set(),
            operators=components.operators,
            objects=objects,
            init_atoms=init_atoms,
            goal_atoms=goal_atoms,
            planner=planner,
        )

    def _build_plan_obs_from_belief(self, belief: Belief | None) -> dict:
        assert self._plan_env is not None
        assert self._plan_env.scene is not None
        assert self.env.scene is not None

        self._plan_env.robot.set_joints(list(self.env.robot.get_joint_positions()))

        held_label = (
            self.env.scene.id_to_label.get(self.env.held_object_id)
            if self.env.held_object_id is not None
            else None
        )
        if held_label is not None:
            self._plan_env.held_object_id = self._plan_env.scene.label_to_id.get(
                held_label
            )
            self._plan_env.grasp_transform = self.env.grasp_transform
        else:
            self._plan_env.held_object_id = None
            self._plan_env.grasp_transform = None

        self._sync_extra_scene_objects()

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

    def _sync_extra_scene_objects(self) -> None:
        """Hook to sync scene objects beyond robot and tracked objects.

        Override in subclasses that have additional scene elements
        (e.g., a target area) that must be kept in sync between the main
        env and the planning env.
        """

    def get_subsequence_effects(
        self, subsequence: list[str]
    ) -> tuple[set[Any], set[Any]]:
        if self.abstractor is None or self._plan_env is None:
            return set(), set()
        components, _ = self._get_planning_components()
        objects_by_name = {
            obj.name.lower(): obj
            for obj in self.abstractor._pybullet_ids  # pylint: disable=protected-access
        }
        return compute_subsequence_effects(
            subsequence, components.operators, objects_by_name
        )

    def is_object_set_unknown(self, obj_names: set[str]) -> bool:
        belief = self.get_belief()
        if belief is None:
            return False
        label_map = self.get_object_labels_for_names(obj_names)
        return any(label in belief.unknown_objects for label in label_map.values())

    def get_object_labels_for_names(self, names: set[str]) -> dict[str, str]:
        """Map from object names in symbolic plans to labels in the
        environment."""
        if self.abstractor is None:
            return {}
        assert self.env.scene is not None
        id_to_label = self.env.scene.id_to_label
        pddl_to_label = {
            obj.name.lower(): id_to_label[obj_id]
            for obj, obj_id in self.abstractor._pybullet_ids.items()  # pylint: disable=protected-access
            if obj_id in id_to_label
        }
        return {name: pddl_to_label[name] for name in names if name in pddl_to_label}

    def plan_for_goal_with_belief(
        self, goal_atoms: Any, belief_override: Any, seed: int | None = None
    ) -> Plan | None:
        """Plan for a goal while temporarily overriding the belief used for
        planning."""
        original_belief = self.env.belief
        self.env.belief = belief_override
        try:
            return self._run_plan(goal_atoms_override=goal_atoms, seed=seed)
        finally:
            self.env.belief = original_belief

    def get_grounded_controller(self, action_str: str):
        """Get a low-level controller in the symbolic plan."""
        if self._last_plan_components is None:
            return None
        tokens = action_str.strip("() ").split()
        if not tokens:
            return None
        op_name, obj_names = tokens[0], tokens[1:]
        components, skills = self._last_plan_components
        skill = next((s for s in skills if s.operator.name == op_name), None)
        if skill is None:
            return None
        objects_by_name = {
            obj.name.lower(): obj
            for obj in components.abstractor._pybullet_ids  # pylint: disable=protected-access
        }
        try:
            objects = [objects_by_name[name.lower()] for name in obj_names]
        except KeyError:
            return None
        return skill.ground(objects).controller

    def get_belief(self) -> Belief | None:
        """Return the current belief state."""
        return self.env.belief

    def is_target_unknown(self, info: dict[str, Any]) -> bool:
        """Check if the target object is currently unknown in the belief."""
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
        """Select an NBV viewpoint for the object to be observed."""
        if self._nbv_planner is None:
            return None
        belief = self.get_belief()
        if belief is None:
            return None
        self._build_plan_obs_from_belief(belief)
        candidate = self._nbv_planner.plan_for_object(
            belief=belief, object_label=target_label, z_range=(0.0, 0.1), seed=seed
        )
        if self.gui and candidate is not None:
            self._draw_nbv_markers(belief, target_label, candidate)
        return candidate

    def _draw_nbv_markers(
        self,
        belief: Belief,
        target_label: str,
        candidate: ViewpointCandidate,
    ) -> None:
        pcid = self.env.physics_client_id
        for item_id in self._nbv_debug_ids:
            p.removeUserDebugItem(item_id, physicsClientId=pcid)
        self._nbv_debug_ids.clear()

        centroid = compute_object_target_center(
            belief, target_label, z_range=(0.0, 0.1)
        )
        vp = np.array(candidate.pose.position)
        r = 0.02

        def _cross(pos: np.ndarray, color: list[float]) -> None:
            for axis in np.eye(3):
                self._nbv_debug_ids.append(
                    p.addUserDebugLine(
                        (pos - r * axis).tolist(),
                        (pos + r * axis).tolist(),
                        color,
                        lineWidth=3,
                        physicsClientId=pcid,
                    )
                )

        _cross(vp, [0.0, 0.0, 1.0])
        if centroid is not None:
            _cross(centroid, [1.0, 0.0, 0.0])
            self._nbv_debug_ids.append(
                p.addUserDebugLine(
                    centroid.tolist(),
                    vp.tolist(),
                    [0.0, 1.0, 0.0],
                    lineWidth=2,
                    physicsClientId=pcid,
                )
            )

    def plan_to_viewpoint(self, viewpoint: ViewpointCandidate) -> list[np.ndarray]:
        """Plan a collision-free trajectory to reach the given viewpoint."""
        self._build_plan_obs_from_belief(self.env.belief)
        assert self._plan_env is not None
        assert self._plan_env.scene is not None

        ee_pose = viewpoint.pose
        if (
            self._nbv_planner is not None
            and self._nbv_planner.camera_to_ee_transform is not None
        ):
            ee_pose = multiply_poses(
                viewpoint.pose, invert_pose(self._nbv_planner.camera_to_ee_transform)
            )

        collision_ids = set(self._plan_env.scene.object_ids)
        if self._plan_env.held_object_id is not None:
            collision_ids.discard(self._plan_env.held_object_id)
        plan = run_smooth_motion_planning_to_pose_with_surface_check(
            ee_pose,
            self._plan_env.robot,
            collision_ids=collision_ids,
            surface_id=self._plan_env.scene.table_id,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed or 0,
            max_candidate_plans=1,
            birrt_num_attempts=10,
            birrt_num_iters=200,
        )

        if plan is None:
            print("[plan_to_viewpoint] Motion planning failed for viewpoint")
            return []

        return [np.array(joints[:7]) for joints in plan]

    def action_from_trajectory(
        self, current_waypoint: np.ndarray, next_waypoint: np.ndarray
    ) -> np.ndarray:
        """Compute a low-level action to move from current_waypoint to
        next_waypoint."""
        joint_delta = next_waypoint - current_waypoint
        action = np.concatenate([joint_delta, [0.0]]).astype(np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)  # type: ignore[attr-defined] # pylint: disable=line-too-long

    def is_viewpoint_reached(
        self, viewpoint: ViewpointCandidate, threshold: float = 0.05
    ) -> bool:
        """Check if the robot has reached the target viewpoint."""
        current_joints = np.array(self.env.robot.get_joint_positions())
        target_joints = np.array(viewpoint.joint_config)
        return float(np.linalg.norm(target_joints[:7] - current_joints[:7])) < threshold

    def get_viewpoint_info_gain(self, viewpoint: ViewpointCandidate) -> float:
        """Get the expected information gain for a viewpoint."""
        return float(viewpoint.expected_ig)

    def get_ground_truth_object_poses(self, obj_ids: list[int]) -> dict[int, tuple]:
        """Get ground truth poses for the given object IDs."""
        poses = {}
        for obj_id in obj_ids:
            pos, quat = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self.env.physics_client_id
            )
            poses[obj_id] = pos + quat
        return poses

    def close(self) -> None:
        """Clean up resources."""
        if self._plan_env is not None:
            self._plan_env.close()
            self._plan_env = None
        self.env.close()
