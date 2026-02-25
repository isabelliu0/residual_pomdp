"""TAMP components for TabletopPickEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

from residual_controllers.envs.tabletop_manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from residual_controllers.geometry import get_half_extents_from_aabb
from residual_controllers.tamp.structs import (
    BeliefAbstractor,
    PredicateContainer,
    TypeContainer,
)

if TYPE_CHECKING:
    from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


@dataclass
class TabletopTypes(TypeContainer):
    """Container for TabletopPickEnv types."""

    def __init__(self) -> None:
        self.robot = Type("robot")
        self.obj = Type("obj")

    def as_set(self) -> set[Type]:
        return {self.robot, self.obj}


@dataclass
class TabletopPredicates(PredicateContainer):
    """Container for TabletopPickEnv predicates."""

    def __init__(self, types: TabletopTypes) -> None:
        self.holding = Predicate("holding", [types.robot, types.obj])
        self.on = Predicate("on", [types.obj, types.obj])
        self.gripper_empty = Predicate("gripper-empty", [types.robot])
        self.is_movable = Predicate("is-movable", [types.obj])
        self.is_target = Predicate("is-target", [types.obj])
        self.is_target_area = Predicate("is-target-area", [types.obj])
        self.in_target_area = Predicate("in-target-area", [types.obj, types.obj])

    def __getitem__(self, key: str) -> Predicate:
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        return {
            self.holding,
            self.on,
            self.gripper_empty,
            self.is_movable,
            self.is_target,
            self.is_target_area,
            self.in_target_area,
        }


class TabletopAbstractor(BeliefAbstractor[dict]):
    """Abstractor for TabletopPickEnv."""

    def __init__(
        self,
        env: TabletopPickEnv,
        types: TabletopTypes,
        predicates: TabletopPredicates,
    ):
        self.env = env
        self.types = types
        self.predicates = predicates
        self._robot_obj = Object("robot", types.robot)
        self._table_obj = Object("table", types.obj)
        self._target_area_obj = Object("target_area", types.obj)
        self._block_objs: list[Object] = []
        self._pybullet_ids: dict[Object, int] = {}
        self._target_id: int | None = None

    def reset(self, obs: dict) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        assert self.env.scene is not None
        assert self.env.robot is not None

        self._block_objs = []
        self._pybullet_ids = {
            self._robot_obj: self.env.robot.robot_id,
            self._table_obj: self.env.scene.table_id,
            self._target_area_obj: self.env.scene.target_area_id,
        }
        for i, obj_id in enumerate(self.env.scene.object_ids):
            label = chr(65 + i)
            block = Object(label, self.types.obj)
            self._block_objs.append(block)
            self._pybullet_ids[block] = obj_id

        self._target_id = self.env.scene.object_ids[self.env.scene.target_idx]
        objects = {self._robot_obj, self._table_obj, self._target_area_obj} | set(
            self._block_objs
        )
        init_atoms = self._get_atoms_from_obs(obs)

        pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
        target_obj = pybullet_id_to_obj[self._target_id]
        goal_atoms = {
            GroundAtom(
                self.predicates.in_target_area, [target_obj, self._target_area_obj]
            )
        }

        return objects, init_atoms, goal_atoms

    def step(self, obs: dict) -> RelationalAbstractState:
        atoms = self._get_atoms_from_obs(obs)
        objects = {self._robot_obj, self._table_obj, self._target_area_obj} | set(
            self._block_objs
        )
        return RelationalAbstractState(atoms=atoms, objects=objects)

    def _get_on_relations(self, held_id: int | None) -> set[tuple[Object, Object]]:
        """Return (upper, lower) pairs where upper is resting on lower."""
        pcid = self.env.physics_client_id
        upper_candidates = [self._target_area_obj] + self._block_objs
        lower_candidates = [self._target_area_obj, self._table_obj] + self._block_objs

        on_relations: set[tuple[Object, Object]] = set()
        on_target_area: set[Object] = set()
        for obj1 in upper_candidates:
            obj1_id = self._pybullet_ids[obj1]
            if obj1_id == held_id:
                continue
            pose1 = get_pose(obj1_id, pcid)
            half1 = get_half_extents_from_aabb(obj1_id, pcid)
            obj1_bottom_z = pose1.position[2] - half1[2]

            for obj2 in lower_candidates:
                if obj1 == obj2:
                    continue
                if obj2 == self._table_obj and obj1 in on_target_area:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                pose2 = get_pose(obj2_id, pcid)
                half2 = get_half_extents_from_aabb(obj2_id, pcid)
                obj2_top_z = pose2.position[2] + half2[2]

                if abs(obj1_bottom_z - obj2_top_z) < 0.005 and check_body_collisions(
                    obj1_id, obj2_id, pcid, distance_threshold=0.002
                ):
                    on_relations.add((obj1, obj2))
                    if obj2 == self._target_area_obj and obj1 in self._block_objs:
                        on_target_area.add(obj1)

        return on_relations

    def _get_atoms_from_obs(self, obs: dict) -> set[GroundAtom]:
        assert self.env.scene is not None
        atoms: set[GroundAtom] = set()
        atoms.add(GroundAtom(self.predicates.is_movable, [self._table_obj]))
        atoms.add(GroundAtom(self.predicates.is_target_area, [self._target_area_obj]))

        held_idx = int(obs["held_object_idx"][0])
        held_id = None
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]

        if held_id is None:
            atoms.add(GroundAtom(self.predicates.gripper_empty, [self._robot_obj]))

        for block in self._block_objs:
            atoms.add(GroundAtom(self.predicates.is_movable, [block]))
            block_id = self._pybullet_ids[block]
            if block_id == self._target_id:
                atoms.add(GroundAtom(self.predicates.is_target, [block]))
            if held_id == block_id:
                atoms.add(GroundAtom(self.predicates.holding, [self._robot_obj, block]))

        for obj1, obj2 in self._get_on_relations(held_id):
            if obj2 == self._target_area_obj and obj1 in self._block_objs:
                atoms.add(GroundAtom(self.predicates.in_target_area, [obj1, obj2]))
            else:
                atoms.add(GroundAtom(self.predicates.on, [obj1, obj2]))

        return atoms

    def get_pybullet_id(self, obj: Object) -> int:
        """Get the PyBullet ID corresponding to an abstract object."""
        return self._pybullet_ids[obj]


def create_tabletop_operators(
    types: TabletopTypes, predicates: TabletopPredicates
) -> set[LiftedOperator]:
    """Create lifted operators for TabletopPickEnv."""
    robot = Variable("?robot", types.robot)
    obj = Variable("?obj", types.obj)
    surface = Variable("?surface", types.obj)

    pick_op = LiftedOperator(
        "pick",
        [robot, obj, surface],
        preconditions={
            LiftedAtom(predicates.gripper_empty, [robot]),
            LiftedAtom(predicates.on, [obj, surface]),
            LiftedAtom(predicates.is_movable, [obj]),
        },
        add_effects={
            LiftedAtom(predicates.holding, [robot, obj]),
        },
        delete_effects={
            LiftedAtom(predicates.gripper_empty, [robot]),
            LiftedAtom(predicates.on, [obj, surface]),
        },
    )

    robot2 = Variable("?robot", types.robot)
    obj2 = Variable("?obj", types.obj)
    target_area = Variable("?target_area", types.obj)

    place_op = LiftedOperator(
        "place",
        [robot2, obj2, target_area],
        preconditions={
            LiftedAtom(predicates.holding, [robot2, obj2]),
            LiftedAtom(predicates.is_target_area, [target_area]),
        },
        add_effects={
            LiftedAtom(predicates.in_target_area, [obj2, target_area]),
            LiftedAtom(predicates.gripper_empty, [robot2]),
        },
        delete_effects={
            LiftedAtom(predicates.holding, [robot2, obj2]),
        },
    )

    return {pick_op, place_op}


class PickGroundController(GroundParameterizedController[dict, np.ndarray]):
    """Controller for picking an object using kinematic planning."""

    def __init__(
        self,
        objects: Sequence[Object],
        env: TabletopPickEnv,
        abstractor: TabletopAbstractor,
    ):
        super().__init__(objects)
        self.env = env
        self.abstractor = abstractor
        self._kinematic_plan: list[KinematicState] | None = None
        self._action_idx = 0
        self._terminated = False
        self._current_params: dict[str, float] | None = None
        self._current_obs: dict | None = None

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._current_params = params
        self._current_obs = x
        self._kinematic_plan = self._compute_kinematic_plan()
        if self._kinematic_plan is None:
            raise TrajectorySamplingFailure("Failed to compute pick plan")

    def terminated(self) -> bool:
        return self._terminated

    def step(self) -> np.ndarray:
        if (
            self._kinematic_plan is None
            or self._action_idx >= len(self._kinematic_plan) - 1
        ):
            self._terminated = True
            return np.zeros(8, dtype=np.float32)

        s0 = self._kinematic_plan[self._action_idx]
        s1 = self._kinematic_plan[self._action_idx + 1]
        self._action_idx += 1

        if self._action_idx >= len(self._kinematic_plan) - 1:
            self._terminated = True

        joint_delta = np.subtract(s1.robot_joints[:7], s0.robot_joints[:7])

        if s1.attachments and not s0.attachments:
            gripper_action = -1.0
        elif s0.attachments and not s1.attachments:
            gripper_action = 1.0
        else:
            gripper_action = 0.0

        action = np.hstack([joint_delta, [gripper_action]]).astype(np.float32)
        return action

    def observe(self, x: dict) -> None:
        pass

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None
        assert self._current_obs is not None

        _, obj, surface = self.objects
        object_id = self.abstractor.get_pybullet_id(obj)
        surface_id = self.abstractor.get_pybullet_id(surface)

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
            pos = tuple(obs_poses[i, :3])
            orn = tuple(obs_poses[i, 3:])
            object_poses[obj_id] = Pose(position=pos, orientation=orn)

        held_idx = int(self._current_obs["held_object_idx"][0])
        attachments: dict[int, Pose] = {}
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
            grasp_tf = self._current_obs["grasp_transform"]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        initial_state = KinematicState(robot_joints, object_poses, attachments)

        initial_state.set_pybullet(self.env.robot)

        collision_ids = set(object_poses.keys())
        plan = get_kinematic_plan_to_pick_object(
            initial_state,
            self.env.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=iter([self._get_grasp_pose()]),
            grasp_generator_iters=5,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
        )

        initial_state.set_pybullet(self.env.robot)

        return plan

    def _get_grasp_pose(self) -> Pose:
        assert self.env.robot is not None
        ee_pose = self.env.robot.get_end_effector_pose()
        _, _, ee_yaw = p.getEulerFromQuaternion(ee_pose.orientation)
        base_orientation = p.getQuaternionFromEuler([np.pi, 0, ee_yaw])
        return Pose((0, 0, 0), base_orientation)


GRASP_PARAMS_SPACE = Box(
    low=np.array([], dtype=np.float32),
    high=np.array([], dtype=np.float32),
    shape=(0,),
)

PLACE_PARAMS_SPACE = Box(
    low=np.array([], dtype=np.float32),
    high=np.array([], dtype=np.float32),
    shape=(0,),
)


class PlaceGroundController(GroundParameterizedController[dict, np.ndarray]):
    """Controller for placing a held object at the target area."""

    def __init__(
        self,
        objects: Sequence[Object],
        env: TabletopPickEnv,
        abstractor: TabletopAbstractor,
    ):
        super().__init__(objects)
        self.env = env
        self.abstractor = abstractor
        self._kinematic_plan: list[KinematicState] | None = None
        self._action_idx = 0
        self._terminated = False
        self._current_params: dict[str, float] | None = None
        self._current_obs: dict | None = None

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._current_params = params
        self._current_obs = x
        self._kinematic_plan = self._compute_kinematic_plan()
        if self._kinematic_plan is None:
            raise TrajectorySamplingFailure("Failed to compute place plan")

    def terminated(self) -> bool:
        return self._terminated

    def step(self) -> np.ndarray:
        if (
            self._kinematic_plan is None
            or self._action_idx >= len(self._kinematic_plan) - 1
        ):
            self._terminated = True
            return np.zeros(8, dtype=np.float32)

        s0 = self._kinematic_plan[self._action_idx]
        s1 = self._kinematic_plan[self._action_idx + 1]
        self._action_idx += 1

        if self._action_idx >= len(self._kinematic_plan) - 1:
            self._terminated = True

        joint_delta = np.subtract(s1.robot_joints[:7], s0.robot_joints[:7])

        if s0.attachments and not s1.attachments:
            gripper_action = 1.0
        else:
            gripper_action = 0.0

        action = np.hstack([joint_delta, [gripper_action]]).astype(np.float32)
        return action

    def observe(self, x: dict) -> None:
        pass

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None
        assert self._current_obs is not None

        _, obj, _ = self.objects
        object_id = self.abstractor.get_pybullet_id(obj)

        target_area_pose_arr = self._current_obs["target_area_pose"]
        area_x, area_y = float(target_area_pose_arr[0]), float(target_area_pose_arr[1])

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
            pos = tuple(obs_poses[i, :3])
            orn = tuple(obs_poses[i, 3:])
            object_poses[obj_id] = Pose(position=pos, orientation=orn)

        held_idx = int(self._current_obs["held_object_idx"][0])
        attachments: dict[int, Pose] = {}
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
            grasp_tf = self._current_obs["grasp_transform"]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        initial_state = KinematicState(robot_joints, object_poses, attachments)
        initial_state.set_pybullet(self.env.robot)

        # Compute desired world pose of the object at the target area center.
        place_z = 0.025  # block half_extents
        world_obj_pose = Pose(
            position=(area_x, area_y, place_z), orientation=(0, 0, 0, 1)
        )

        # Express placement in the table frame.
        table_pose = get_pose(self.env.scene.table_id, self.env.physics_client_id)
        relative_placement = multiply_poses(table_pose.invert(), world_obj_pose)

        def placement_gen():
            yield relative_placement

        collision_ids = set(object_poses.keys())
        plan = get_kinematic_plan_to_place_object(
            initial_state,
            self.env.robot,
            object_id,
            self.env.scene.table_id,
            collision_ids,
            placement_generator=placement_gen(),  # type: ignore[no-untyped-call]
            placement_generator_iters=1,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
            retract_after=False,
        )

        initial_state.set_pybullet(self.env.robot)
        return plan


def create_tabletop_skills(
    types: TabletopTypes,
    operators: set[LiftedOperator],
    env: TabletopPickEnv,
    abstractor: TabletopAbstractor,
) -> set[LiftedSkill]:
    """Create lifted skills for TabletopPickEnv."""
    robot_var = Variable("?robot", types.robot)
    obj_var = Variable("?obj", types.obj)
    surface_var = Variable("?surface", types.obj)
    target_area_var = Variable("?target_area", types.obj)

    pick_operator = next(op for op in operators if op.name == "pick")
    place_operator = next(op for op in operators if op.name == "place")

    def pick_controller_factory(env: TabletopPickEnv, abstractor: TabletopAbstractor):
        return lambda objects: PickGroundController(objects, env, abstractor)

    def place_controller_factory(env: TabletopPickEnv, abstractor: TabletopAbstractor):
        return lambda objects: PlaceGroundController(objects, env, abstractor)

    pick_lifted_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            variables=[robot_var, obj_var, surface_var],
            controller_cls=pick_controller_factory(env, abstractor),
            params_space=GRASP_PARAMS_SPACE,
        )
    )

    place_lifted_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            variables=[robot_var, obj_var, target_area_var],
            controller_cls=place_controller_factory(env, abstractor),
            params_space=PLACE_PARAMS_SPACE,
        )
    )

    pick_skill = LiftedSkill(operator=pick_operator, controller=pick_lifted_controller)
    place_skill = LiftedSkill(
        operator=place_operator, controller=place_lifted_controller
    )

    return {pick_skill, place_skill}
