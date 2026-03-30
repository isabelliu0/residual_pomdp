"""TAMP components for NutAssemblyEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedOperator,
    LiftedParameterizedController,
    LiftedSkill,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    Predicate,
    Variable,
)

from residual_controllers.envs.nut_assembly_env import (
    _NUT_HALF_HEIGHT,
    NutAssemblyScene,
)
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_tamp_base import (
    GRASP_PARAMS_SPACE,
    ObjPickGroundController,
    TabletopBaseAbstractor,
    TabletopBasePredicates,
    TabletopTypes,
    create_pick_operator,
)
from residual_controllers.tamp.collision_utils import (
    run_smooth_motion_planning_to_pose_with_surface_check,
)


@dataclass
class NutAssemblyPredicates(TabletopBasePredicates):
    """Predicates for NutAssemblyEnv."""

    def __init__(self, types: TabletopTypes) -> None:
        super().__init__(types)
        self.is_peg = Predicate("is-peg", [types.obj])
        self.assembled = Predicate("assembled", [types.robot, types.obj, types.obj])

    def as_set(self) -> set[Predicate]:
        return super().as_set() | {self.is_peg, self.assembled}


class NutAssemblyAbstractor(TabletopBaseAbstractor):
    """Abstractor for NutAssemblyEnv."""

    def __init__(
        self,
        env: object,
        types: TabletopTypes,
        predicates: NutAssemblyPredicates,
    ) -> None:
        assert isinstance(env, TabletopBaseEnv)
        super().__init__(env, types, predicates)
        self._nut_predicates = predicates
        self._nut_obj: Object | None = None
        self._peg_obj: Object | None = None

    def _setup_extra_objects(self) -> None:
        scene = self.env.scene
        assert isinstance(scene, NutAssemblyScene)
        self._nut_obj = self._movable_objs[scene.nut_idx]
        self._peg_obj = self._movable_objs[scene.peg_idx]

    def _get_goal_atoms(self) -> set[GroundAtom]:
        assert self._nut_obj is not None and self._peg_obj is not None
        return {
            GroundAtom(
                self._nut_predicates.assembled,
                [self._robot_obj, self._nut_obj, self._peg_obj],
            )
        }

    def _get_atoms_from_obs(self, obs: dict) -> set[GroundAtom]:
        atoms = super()._get_atoms_from_obs(obs)

        if self._peg_obj is not None:
            atoms.add(GroundAtom(self._nut_predicates.is_peg, [self._peg_obj]))

        if self._nut_obj is not None and self._peg_obj is not None:
            assert self.env.scene is not None
            held_idx = int(obs["held_object_idx"][0])
            held_id = None
            if held_idx >= 0:
                held_id = self.env.scene.object_ids[held_idx]  # type: ignore[union-attr]

            nut_id = self._pybullet_ids[self._nut_obj]
            peg_id = self._pybullet_ids[self._peg_obj]

            if held_id != nut_id:
                object_ids = self.env.scene.object_ids  # type: ignore[union-attr]
                nut_scene_idx = object_ids.index(nut_id)
                peg_scene_idx = object_ids.index(peg_id)
                obs_poses = obs["object_poses"]
                nut_pos = obs_poses[nut_scene_idx, :3]
                peg_pos = obs_poses[peg_scene_idx, :3]
                peg_cx = float(peg_pos[0])
                peg_cy = float(peg_pos[1])
                xy_dist = float(
                    np.linalg.norm(nut_pos[:2] - np.array([peg_cx, peg_cy]))
                )
                if xy_dist < 1e-3 and abs(nut_pos[2] - _NUT_HALF_HEIGHT) < 1e-3:
                    atoms.add(
                        GroundAtom(
                            self._nut_predicates.assembled,
                            [self._robot_obj, self._nut_obj, self._peg_obj],
                        )
                    )
        return atoms


ASSEMBLE_PARAMS_SPACE = Box(
    low=np.array([], dtype=np.float32),
    high=np.array([], dtype=np.float32),
    shape=(0,),
)


class AssembleGroundController(GroundParameterizedController[dict, np.ndarray]):
    """Controller that positions the held nut above the peg and lowers it down
    the shaft.

    Moves to a pre-assembly pose high above the peg center, then
    descends straight down along the peg axis so the nut slides onto the
    peg. The peg is excluded from collision checking to allow the nut to
    pass around it.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        env: object,
        abstractor: NutAssemblyAbstractor,
    ):
        assert isinstance(env, TabletopBaseEnv)
        super().__init__(objects)
        self.env = env
        self.abstractor = abstractor
        self._kinematic_plan: list[KinematicState] | None = None
        self._action_idx = 0
        self._terminated = False
        self._current_obs: dict | None = None

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._current_obs = x
        self._kinematic_plan = self._compute_kinematic_plan()
        if self._kinematic_plan is None:
            raise TrajectorySamplingFailure("Failed to compute assemble plan")

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
        gripper_action = 1.0 if (s0.attachments and not s1.attachments) else 0.0
        return np.hstack([joint_delta, [gripper_action]]).astype(np.float32)

    def observe(self, x: dict) -> None:
        pass

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None
        assert self._current_obs is not None

        _, nut_obj, peg_obj, _ = self.objects
        nut_id = self.abstractor.get_pybullet_id(nut_obj)
        peg_id = self.abstractor.get_pybullet_id(peg_obj)
        table_id = self.env.scene.table_id  # type: ignore[union-attr]

        robot_joints = list(self._current_obs["joint_positions"])
        if len(robot_joints) >= 9:
            finger_avg = (robot_joints[7] + robot_joints[8]) / 2
            robot_joints[7] = finger_avg
            robot_joints[8] = finger_avg

        object_poses: dict[int, Pose] = {
            table_id: Pose(position=(0.5, 0.0, -0.015)),
        }
        obs_poses = self._current_obs["object_poses"]
        for i, obj_id in enumerate(self.env.scene.object_ids):  # type: ignore[union-attr]  # pylint: disable=line-too-long
            pos = tuple(obs_poses[i, :3])
            orn = tuple(obs_poses[i, 3:])
            object_poses[obj_id] = Pose(position=pos, orientation=orn)

        held_idx = int(self._current_obs["held_object_idx"][0])
        attachments: dict[int, Pose] = {}
        held_id = None
        grasp_tf = self._current_obs["grasp_transform"]
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]  # type: ignore[union-attr]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        initial_state = KinematicState(robot_joints, object_poses, attachments)
        initial_state.set_pybullet(self.env.robot)
        state = initial_state
        plan = [state]

        peg_pose = object_poses[peg_id]
        peg_cx = float(peg_pose.position[0])
        peg_cy = float(peg_pose.position[1])
        peg_top_z = float(
            p.getAABB(peg_id, physicsClientId=self.env.physics_client_id)[1][2]
        )

        nut_attachment = attachments[nut_id]
        collision_ids = set(object_poses.keys()) - {nut_id, table_id}

        curr_ee = self.env.robot.get_end_effector_pose()
        _, _, ee_yaw = p.getEulerFromQuaternion(curr_ee.orientation)
        down_quat = p.getQuaternionFromEuler([np.pi, 0, ee_yaw])

        nut_world_z = float(multiply_poses(curr_ee, nut_attachment).position[2])
        grasp_z_offset = float(curr_ee.position[2]) - nut_world_z
        descent_ee_z = peg_top_z - grasp_z_offset - 0.015

        # Approach: BiRRT motion plan with peg in collision_ids.
        approach_pose = Pose((peg_cx, peg_cy, peg_top_z + 0.05), down_quat)
        state.set_pybullet(self.env.robot)
        motion_plan = run_smooth_motion_planning_to_pose_with_surface_check(
            approach_pose,
            self.env.robot,
            collision_ids=collision_ids,
            surface_id=table_id,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=0,
            max_time=2.0,
            held_object=nut_id,
            base_link_to_held_obj=nut_attachment,
        )
        if motion_plan is None:
            return None
        for robot_joints_step in motion_plan:
            state = state.copy_with(robot_joints=robot_joints_step)
            plan.append(state)

        # Descent: straight-line Cartesian path down along peg axis.
        state.set_pybullet(self.env.robot)
        curr_ee = self.env.robot.get_end_effector_pose()
        fixed_x = float(curr_ee.position[0])
        fixed_y = float(curr_ee.position[1])
        start_z = float(curr_ee.position[2])

        for z in np.linspace(start_z, descent_ee_z, 21)[1:]:
            ik = p.calculateInverseKinematics(
                self.env.robot.robot_id,
                self.env.robot.end_effector_id,
                (fixed_x, fixed_y, z),
                targetOrientation=down_quat,
                maxNumIterations=1000,
                residualThreshold=1e-6,
                physicsClientId=self.env.physics_client_id,
            )
            new_joints = list(ik[:7]) + list(state.robot_joints[7:])
            state = state.copy_with(robot_joints=new_joints)
            state.set_pybullet(self.env.robot)
            plan.append(state)

        release_state = KinematicState(state.robot_joints, state.object_poses, {})
        for _ in range(50):
            plan.append(release_state)

        initial_state.set_pybullet(self.env.robot)
        if held_idx >= 0:
            self.env.held_object_id = held_id
            self.env.grasp_transform = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )
        else:
            self.env.held_object_id = None
            self.env.grasp_transform = None

        return plan


def create_nut_assembly_operators(
    types: TabletopTypes, predicates: NutAssemblyPredicates
) -> set[LiftedOperator]:
    """Create LiftedOperators for NutAssemblyEnv."""
    pick_op = create_pick_operator(types, predicates)

    robot = Variable("?robot", types.robot)
    nut = Variable("?nut", types.obj)
    peg = Variable("?peg", types.obj)
    surface = Variable("?surface", types.obj)

    assemble_op = LiftedOperator(
        "assemble",
        [robot, nut, peg, surface],
        preconditions={
            LiftedAtom(predicates.holding, [robot, nut]),
            LiftedAtom(predicates.is_peg, [peg]),
            LiftedAtom(predicates.on, [peg, surface]),
        },
        add_effects={
            LiftedAtom(predicates.assembled, [robot, nut, peg]),
            LiftedAtom(predicates.gripper_empty, [robot]),
            LiftedAtom(predicates.on, [nut, surface]),
        },
        delete_effects={
            LiftedAtom(predicates.holding, [robot, nut]),
        },
    )

    return {pick_op, assemble_op}


def create_nut_assembly_skills(
    types: TabletopTypes,
    operators: set[LiftedOperator],
    env: object,
    abstractor: NutAssemblyAbstractor,
) -> set[LiftedSkill]:
    """Create LiftedSkills for NutAssemblyEnv."""
    assert isinstance(env, TabletopBaseEnv)

    robot_var = Variable("?robot", types.robot)
    obj_var = Variable("?obj", types.obj)
    nut_var = Variable("?nut", types.obj)
    peg_var = Variable("?peg", types.obj)
    surface_var = Variable("?surface", types.obj)

    pick_operator = next(op for op in operators if op.name == "pick")
    assemble_operator = next(op for op in operators if op.name == "assemble")

    def _pick_factory(e: TabletopBaseEnv, a: NutAssemblyAbstractor) -> Any:
        return lambda objects: ObjPickGroundController(objects, e, a)

    pick_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, obj_var, surface_var],
        controller_cls=_pick_factory(env, abstractor),
        params_space=GRASP_PARAMS_SPACE,
    )
    pick_skill = LiftedSkill(operator=pick_operator, controller=pick_lifted)

    def _assemble_factory(e: TabletopBaseEnv, a: NutAssemblyAbstractor) -> Any:
        return lambda objects: AssembleGroundController(objects, e, a)

    assemble_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, nut_var, peg_var, surface_var],
        controller_cls=_assemble_factory(env, abstractor),
        params_space=ASSEMBLE_PARAMS_SPACE,
    )
    assemble_skill = LiftedSkill(operator=assemble_operator, controller=assemble_lifted)

    return {pick_skill, assemble_skill}
