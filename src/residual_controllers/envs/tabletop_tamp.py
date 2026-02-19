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
from pybullet_helpers.geometry import Pose, get_pose
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
)
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

    def __getitem__(self, key: str) -> Predicate:
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        return {
            self.holding,
            self.on,
            self.gripper_empty,
            self.is_movable,
            self.is_target,
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
        }
        for i, obj_id in enumerate(self.env.scene.object_ids):
            label = chr(65 + i)
            block = Object(label, self.types.obj)
            self._block_objs.append(block)
            self._pybullet_ids[block] = obj_id

        self._target_id = self.env.scene.object_ids[self.env.scene.target_idx]
        objects = {self._robot_obj, self._table_obj} | set(self._block_objs)
        init_atoms = self._get_atoms()

        pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
        target_obj = pybullet_id_to_obj[self._target_id]
        goal_atoms = {
            GroundAtom(self.predicates.holding, [self._robot_obj, target_obj])
        }

        return objects, init_atoms, goal_atoms

    def step(self, obs: dict) -> RelationalAbstractState:
        atoms = self._get_atoms()
        objects = {self._robot_obj, self._table_obj} | set(self._block_objs)
        return RelationalAbstractState(atoms=atoms, objects=objects)

    def _get_atoms(self) -> set[GroundAtom]:
        atoms: set[GroundAtom] = set()
        atoms.add(GroundAtom(self.predicates.is_movable, [self._table_obj]))

        held_id = self._get_held_object_id()
        if held_id is None:
            atoms.add(GroundAtom(self.predicates.gripper_empty, [self._robot_obj]))

        for block in self._block_objs:
            atoms.add(GroundAtom(self.predicates.is_movable, [block]))

            block_id = self._pybullet_ids[block]
            if block_id == self._target_id:
                atoms.add(GroundAtom(self.predicates.is_target, [block]))

            if held_id == block_id:
                atoms.add(GroundAtom(self.predicates.holding, [self._robot_obj, block]))
            else:
                atoms.add(GroundAtom(self.predicates.on, [block, self._table_obj]))

        return atoms

    def _get_held_object_id(self) -> int | None:
        assert self.env.robot is not None
        gripper_joints = self.env.robot.get_joint_positions()[7:]
        if not all(j < 0.02 for j in gripper_joints):
            return None

        ee_pose = self.env.robot.get_end_effector_pose()
        for block in self._block_objs:
            block_id = self._pybullet_ids[block]
            block_pose = get_pose(block_id, self.env.physics_client_id)
            dist = np.sqrt(
                (ee_pose.position[0] - block_pose.position[0]) ** 2
                + (ee_pose.position[1] - block_pose.position[1]) ** 2
                + (ee_pose.position[2] - block_pose.position[2]) ** 2
            )
            if dist < 0.06:
                return block_id
        return None

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

    return {pick_op}


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

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._current_params = params
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
            return np.zeros(7, dtype=np.float32)

        s0 = self._kinematic_plan[self._action_idx]
        s1 = self._kinematic_plan[self._action_idx + 1]
        self._action_idx += 1

        if self._action_idx >= len(self._kinematic_plan) - 1:
            self._terminated = True

        joint_delta = np.subtract(s1.robot_joints[:7], s0.robot_joints[:7])
        return joint_delta.astype(np.float32)

    def observe(self, x: dict) -> None:
        pass

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None

        _, obj, surface = self.objects
        object_id = self.abstractor.get_pybullet_id(obj)
        surface_id = self.abstractor.get_pybullet_id(surface)

        robot_joints = list(self.env.robot.get_joint_positions())
        object_poses: dict[int, Pose] = {
            self.env.scene.table_id: Pose(position=(0.5, 0.0, -0.015)),
        }
        for obj_id in self.env.scene.object_ids:
            object_poses[obj_id] = get_pose(obj_id, self.env.physics_client_id)

        state = KinematicState(robot_joints, object_poses, {})
        collision_ids = set(object_poses.keys())
        return get_kinematic_plan_to_pick_object(
            state,
            self.env.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=iter([self._get_grasp_pose()]),
            grasp_generator_iters=5,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
        )

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

    pick_operator = next(op for op in operators if op.name == "pick")

    def pick_controller_factory(env: TabletopPickEnv, abstractor: TabletopAbstractor):
        return lambda objects: PickGroundController(objects, env, abstractor)

    pick_lifted_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            variables=[robot_var, obj_var, surface_var],
            controller_cls=pick_controller_factory(env, abstractor),
            params_space=GRASP_PARAMS_SPACE,
        )
    )

    pick_skill = LiftedSkill(operator=pick_operator, controller=pick_lifted_controller)

    return {pick_skill}
