"""Shared TAMP components for tabletop environments."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedOperator,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    Predicate,
    Type,
    Variable,
)

from residual_controllers.beliefs.structs import TabletopState
from residual_controllers.envs.tabletop_manipulation import (
    get_kinematic_plan_to_pick_object,
)
from residual_controllers.geometry import get_half_extents_from_aabb
from residual_controllers.tamp.structs import (
    BeliefAbstractor,
    PredicateContainer,
    TypeContainer,
)

if TYPE_CHECKING:
    from residual_controllers.envs.tabletop_base import TabletopBaseEnv


@dataclass
class TabletopTypes(TypeContainer):
    """Types shared across all tabletop environments."""

    def __init__(self) -> None:
        self.robot = Type("robot")
        self.obj = Type("obj")

    def as_set(self) -> set[Type]:
        return {self.robot, self.obj}


@dataclass
class TabletopBasePredicates(PredicateContainer):
    """Predicates common to all tabletop environments."""

    def __init__(self, types: TabletopTypes) -> None:
        self.holding = Predicate("holding", [types.robot, types.obj])
        self.on = Predicate("on", [types.obj, types.obj])
        self.gripper_empty = Predicate("gripper-empty", [types.robot])
        self.is_movable = Predicate("is-movable", [types.obj])

    def __getitem__(self, key: str) -> Predicate:
        return next(pred for pred in self.as_set() if pred.name == key)

    def as_set(self) -> set[Predicate]:
        return {self.holding, self.on, self.gripper_empty, self.is_movable}


class TabletopBaseAbstractor(BeliefAbstractor[dict]):
    """Base abstractor with common logic for all tabletop environments."""

    def __init__(
        self,
        env: TabletopBaseEnv,
        types: TabletopTypes,
        predicates: TabletopBasePredicates,
    ) -> None:
        self.env = env
        self.types = types
        self.predicates = predicates
        self._robot_obj = Object("robot", types.robot)
        self._table_obj = Object("table", types.obj)
        self._movable_objs: list[Object] = []
        self._pybullet_ids: dict[Object, int] = {}

    def reset(self, obs: dict) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        assert self.env.scene is not None
        assert self.env.robot is not None
        self._movable_objs = []
        self._pybullet_ids = {
            self._robot_obj: self.env.robot.robot_id,
            self._table_obj: self.env.scene.table_id,
        }
        for i, obj_id in enumerate(self.env.scene.object_ids):
            label = chr(65 + i)
            obj = Object(label, self.types.obj)
            self._movable_objs.append(obj)
            self._pybullet_ids[obj] = obj_id
        self._setup_extra_objects()
        objects = self._get_objects()
        init_atoms = self._get_atoms_from_obs(obs)
        goal_atoms = self._get_goal_atoms()
        return objects, init_atoms, goal_atoms

    def _setup_extra_objects(self) -> None:
        """Hook for subclasses to register env-specific objects."""

    def _get_objects(self) -> set[Object]:
        return {self._robot_obj, self._table_obj} | set(self._movable_objs)

    @abstractmethod
    def _get_goal_atoms(self) -> set[GroundAtom]:
        """Return goal atoms specific to this environment."""

    def step(self, obs: dict) -> RelationalAbstractState:
        atoms = self._get_atoms_from_obs(obs)
        objects = self._get_objects()
        return RelationalAbstractState(atoms=atoms, objects=objects)

    def _get_on_relations(self, held_id: int | None) -> set[tuple[Object, Object]]:
        pcid = self.env.physics_client_id
        on_relations: set[tuple[Object, Object]] = set()
        lower_candidates = [self._table_obj] + self._movable_objs
        for obj1 in self._movable_objs:
            obj1_id = self._pybullet_ids[obj1]
            if obj1_id == held_id:
                continue
            pose1 = get_pose(obj1_id, pcid)
            half1 = get_half_extents_from_aabb(obj1_id, pcid)
            obj1_bottom_z = pose1.position[2] - half1[2]
            for obj2 in lower_candidates:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                pose2 = get_pose(obj2_id, pcid)
                half2 = get_half_extents_from_aabb(obj2_id, pcid)
                obj2_top_z = pose2.position[2] + half2[2]
                if abs(obj1_bottom_z - obj2_top_z) >= 0.005:
                    continue
                if check_body_collisions(
                    obj1_id, obj2_id, pcid, distance_threshold=0.002
                ):
                    on_relations.add((obj1, obj2))
        return on_relations

    def _atoms_from_on_relation(self, obj1: Object, obj2: Object) -> set[GroundAtom]:
        """Map an on-relation pair to ground atoms.

        Override to customize.
        """
        return {GroundAtom(self.predicates.on, [obj1, obj2])}

    def _get_atoms_from_obs(self, obs: dict) -> set[GroundAtom]:
        assert self.env.scene is not None
        atoms: set[GroundAtom] = set()
        held_idx = int(obs["held_object_idx"][0])
        held_id = None
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
        if held_id is None:
            atoms.add(GroundAtom(self.predicates.gripper_empty, [self._robot_obj]))
        for obj in self._movable_objs:
            atoms.add(GroundAtom(self.predicates.is_movable, [obj]))
            if self._pybullet_ids[obj] == held_id:
                atoms.add(GroundAtom(self.predicates.holding, [self._robot_obj, obj]))
        for obj1, obj2 in self._get_on_relations(held_id):
            atoms |= self._atoms_from_on_relation(obj1, obj2)
        return atoms

    def get_pybullet_id(self, obj: Object) -> int:
        """Return the PyBullet body ID corresponding to the given object."""
        return self._pybullet_ids[obj]

    def get_atoms_from_belief_particles(
        self,
        particles: list[TabletopState],
        held_obs: dict,
    ) -> set[GroundAtom]:
        """Optimize atom abstraction by taking the union across particles."""
        assert self.env.scene is not None
        union: set[GroundAtom] = set()
        for particle in particles:
            for label, sim_obj_id in self.env.scene.label_to_id.items():
                pose = particle.object_poses.get(label)
                if pose is not None:
                    set_pose(
                        sim_obj_id,
                        Pose(position=pose[:3], orientation=pose[3:]),
                        self.env.physics_client_id,
                    )
            union |= self._get_atoms_from_obs(held_obs)
        return union


def _make_checking_state(obs: dict) -> SimpleNamespace:
    held = int(obs["held_object_idx"][0])
    return SimpleNamespace(
        robot_joints=list(obs["joint_positions"]),
        attachments={0: None} if held >= 0 else {},
    )


GRASP_PARAMS_SPACE = Box(
    low=np.array([], dtype=np.float32),
    high=np.array([], dtype=np.float32),
    shape=(0,),
)


class PickGroundController(GroundParameterizedController[dict, np.ndarray]):
    """Controller for picking an object with PreGrasp / Grasp / PostGrasp
    phases."""

    def __init__(
        self,
        objects: Sequence[Object],
        env: TabletopBaseEnv,
        abstractor: TabletopBaseAbstractor,
    ):
        super().__init__(objects)
        self.env = env
        self.abstractor = abstractor
        self._plan: Sequence[KinematicState | SimpleNamespace] | None = None
        self._attach_idx: int = -1
        self._action_idx = 0
        self._phase = "pre_grasp"
        self._terminated = False
        self._current_obs: dict | None = None

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._phase = "pre_grasp"
        self._current_obs = x
        self._plan = self._compute_kinematic_plan()
        if self._plan is None:
            raise TrajectorySamplingFailure("Failed to compute pick plan")
        self._attach_idx = next(
            (
                i
                for i in range(1, len(self._plan))
                if self._plan[i].attachments and not self._plan[i - 1].attachments
            ),
            -1,
        )
        if self._attach_idx < 0:
            raise TrajectorySamplingFailure("No grasp transition found in pick plan")

    def terminated(self) -> bool:
        return self._terminated

    def observe(self, x: dict) -> None:
        self._current_obs = x

    def step(self) -> np.ndarray:
        if self._phase == "grasp":
            assert self._current_obs is not None
            if int(self._current_obs["held_object_idx"][0]) < 0:
                raise TrajectorySamplingFailure(
                    "Grasp failed: no object held after close"
                )
            self._phase = "post_grasp"

        if self._plan is None or self._action_idx >= len(self._plan) - 1:
            self._terminated = True
            return np.zeros(8, dtype=np.float32)

        s0 = self._plan[self._action_idx]
        s1 = self._plan[self._action_idx + 1]
        self._action_idx += 1

        if self._action_idx == self._attach_idx:
            assert self._current_obs is not None
            assert self.env.scene is not None
            ee_xy = np.array(self._current_obs["camera_pose"][:2], dtype=float)
            _, obj, _ = self.objects
            obj_id = self.abstractor.get_pybullet_id(obj)
            obj_idx = self.env.scene.object_ids.index(obj_id)
            obj_xy = np.array(
                self._current_obs["object_poses"][obj_idx, :2], dtype=float
            )
            if np.linalg.norm(ee_xy - obj_xy) > 0.05:
                raise TrajectorySamplingFailure("PreGrasp failed: EE not above object")
            self._phase = "grasp"

        if self._action_idx >= len(self._plan) - 1:
            self._terminated = True

        joint_delta = np.subtract(s1.robot_joints[:7], s0.robot_joints[:7])
        if s1.attachments and not s0.attachments:
            gripper_action = -1.0
        elif s0.attachments and not s1.attachments:
            gripper_action = 1.0
        else:
            gripper_action = 0.0

        return np.hstack([joint_delta, [gripper_action]]).astype(np.float32)

    def reset_for_checking(self, plan: Any, initial_obs: dict) -> None:
        """Reset for checking with a given plan and initial observation."""
        self._terminated = False
        self._action_idx = 0
        self._phase = "pre_grasp"
        self._current_obs = initial_obs
        states = list(plan.states)
        self._attach_idx = next(
            (
                i
                for i in range(1, len(states))
                if int(states[i - 1]["held_object_idx"][0])
                < 0
                <= int(states[i]["held_object_idx"][0])
            ),
            -1,
        )
        if self._attach_idx < 0:
            raise TrajectorySamplingFailure("No grasp transition found in plan states")
        self._plan = [_make_checking_state(s) for s in states]

    def _build_initial_state_from_obs(self) -> tuple[KinematicState, int]:
        """Build KinematicState from current obs.

        Returns (state, held_idx).
        """
        assert self.env.scene is not None
        assert self._current_obs is not None

        robot_joints = list(self._current_obs["joint_positions"])
        if len(robot_joints) >= 9:
            finger_avg = (robot_joints[7] + robot_joints[8]) / 2
            robot_joints[7] = finger_avg
            robot_joints[8] = finger_avg

        object_poses: dict[int, Pose] = {
            self.env.scene.table_id: Pose(position=(0.5, 0.0, -0.015)),
        }
        obs_poses = self._current_obs["object_poses"]
        for i, obj_id in enumerate(self.env.scene.object_ids):
            object_poses[obj_id] = Pose(
                position=tuple(obs_poses[i, :3]),
                orientation=tuple(obs_poses[i, 3:]),
            )

        held_idx = int(self._current_obs["held_object_idx"][0])
        grasp_tf = self._current_obs["grasp_transform"]
        attachments: dict[int, Pose] = {}
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        return KinematicState(robot_joints, object_poses, attachments), held_idx

    def _restore_env_grip_state(self, held_idx: int) -> None:
        assert self._current_obs is not None
        assert self.env.scene is not None
        grasp_tf = self._current_obs["grasp_transform"]
        if held_idx >= 0:
            self.env.held_object_id = self.env.scene.object_ids[held_idx]
            self.env.grasp_transform = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )
        else:
            self.env.held_object_id = None
            self.env.grasp_transform = None

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None

        _, obj, surface = self.objects
        object_id = self.abstractor.get_pybullet_id(obj)
        surface_id = self.abstractor.get_pybullet_id(surface)

        initial_state, held_idx = self._build_initial_state_from_obs()
        initial_state.set_pybullet(self.env.robot)

        collision_ids = set(initial_state.object_poses.keys())
        plan = get_kinematic_plan_to_pick_object(
            initial_state,
            self.env.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=iter([self._get_grasp_pose()]),
            start_ignored_collision_bodies={object_id},
            grasp_generator_iters=5,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
        )

        initial_state.set_pybullet(self.env.robot)
        self._restore_env_grip_state(held_idx)

        return plan

    def _get_grasp_pose(self) -> Pose:
        assert self.env.robot is not None
        ee_pose = self.env.robot.get_end_effector_pose()
        _, _, ee_yaw = p.getEulerFromQuaternion(ee_pose.orientation)
        base_orientation = p.getQuaternionFromEuler([np.pi, 0, ee_yaw])
        return Pose((0, 0, 0), base_orientation)


def create_pick_operator(
    types: TabletopTypes, predicates: TabletopBasePredicates
) -> LiftedOperator:
    """Create a LiftedOperator for picking an object."""
    robot = Variable("?robot", types.robot)
    obj = Variable("?obj", types.obj)
    surface = Variable("?surface", types.obj)
    return LiftedOperator(
        "pick",
        [robot, obj, surface],
        preconditions={
            LiftedAtom(predicates.gripper_empty, [robot]),
            LiftedAtom(predicates.on, [obj, surface]),
            LiftedAtom(predicates.is_movable, [obj]),
        },
        add_effects={LiftedAtom(predicates.holding, [robot, obj])},
        delete_effects={
            LiftedAtom(predicates.gripper_empty, [robot]),
            LiftedAtom(predicates.on, [obj, surface]),
        },
    )


class ObjPickGroundController(PickGroundController):
    """Pick controller using top-of-AABB world-frame grasp for Objaverse
    objects."""

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None

        _, obj, surface = self.objects
        object_id = self.abstractor.get_pybullet_id(obj)
        surface_id = self.abstractor.get_pybullet_id(surface)

        initial_state, held_idx = self._build_initial_state_from_obs()
        initial_state.set_pybullet(self.env.robot)

        collision_ids = set(initial_state.object_poses.keys())
        plan = get_kinematic_plan_to_pick_object(
            initial_state,
            self.env.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=self._grasp_poses(object_id),
            start_ignored_collision_bodies={object_id},
            grasp_generator_iters=5,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
        )

        initial_state.set_pybullet(self.env.robot)
        self._restore_env_grip_state(held_idx)

        return plan

    def _grasp_poses(self, object_id: int):  # type: ignore[override]
        assert self.env.robot is not None
        aabb = p.getAABB(object_id, physicsClientId=self.env.physics_client_id)
        obj_center_x = (float(aabb[0][0]) + float(aabb[1][0])) / 2
        obj_center_y = (float(aabb[0][1]) + float(aabb[1][1])) / 2
        aabb_top_z = float(aabb[1][2])

        ee_pose = self.env.robot.get_end_effector_pose()
        _, _, ee_yaw = p.getEulerFromQuaternion(ee_pose.orientation)

        obj_pose = get_pose(object_id, self.env.physics_client_id)
        inv_pos, inv_orn = p.invertTransform(obj_pose.position, obj_pose.orientation)
        obj_pose_inv = Pose(position=inv_pos, orientation=inv_orn)

        for z_rot in [ee_yaw, ee_yaw - np.pi / 2]:
            world_grasp_orientation = p.getQuaternionFromEuler([np.pi, 0, z_rot])
            world_grasp = Pose(
                (obj_center_x, obj_center_y, aabb_top_z), world_grasp_orientation
            )
            yield multiply_poses(obj_pose_inv, world_grasp)


def create_pick_skill(
    types: TabletopTypes,
    pick_operator: LiftedOperator,
    env: TabletopBaseEnv,
    abstractor: TabletopBaseAbstractor,
) -> LiftedSkill:
    """Create a LiftedSkill for picking an object."""
    robot_var = Variable("?robot", types.robot)
    obj_var = Variable("?obj", types.obj)
    surface_var = Variable("?surface", types.obj)

    def _pick_factory(e: TabletopBaseEnv, a: TabletopBaseAbstractor) -> Any:
        return lambda objects: PickGroundController(objects, e, a)

    pick_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, obj_var, surface_var],
        controller_cls=_pick_factory(env, abstractor),
        params_space=GRASP_PARAMS_SPACE,
    )
    return LiftedSkill(operator=pick_operator, controller=pick_lifted)
