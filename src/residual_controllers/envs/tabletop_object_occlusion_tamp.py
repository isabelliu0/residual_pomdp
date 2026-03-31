"""TAMP components for TabletopObjectOcclusionEnv (pour task)."""

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
from pybullet_helpers.geometry import Pose, get_pose, iter_between_poses, multiply_poses
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    Predicate,
    Variable,
)

from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_object_occlusion import (
    TabletopObjectOcclusionScene,
)
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
class TabletopPourPredicates(TabletopBasePredicates):
    """Predicates for TabletopObjectOcclusionEnv."""

    def __init__(self, types: TabletopTypes) -> None:
        super().__init__(types)
        self.is_cup = Predicate("is-cup", [types.obj])
        self.pouring = Predicate("pouring", [types.robot, types.obj, types.obj])

    def as_set(self) -> set[Predicate]:
        return super().as_set() | {self.is_cup, self.pouring}


class TabletopPourAbstractor(TabletopBaseAbstractor):
    """Abstractor for TabletopObjectOcclusionEnv."""

    def __init__(
        self,
        env: object,
        types: TabletopTypes,
        predicates: TabletopPourPredicates,
    ) -> None:
        assert isinstance(env, TabletopBaseEnv)
        super().__init__(env, types, predicates)
        self._pour_predicates = predicates
        self._milk_obj: Object | None = None
        self._cup_obj: Object | None = None

    def _setup_extra_objects(self) -> None:
        scene = self.env.scene
        assert isinstance(scene, TabletopObjectOcclusionScene)
        self._milk_obj = self._movable_objs[scene.milk_carton_idx]
        self._cup_obj = self._movable_objs[scene.cup_idx]

    def _get_on_relations(self, held_id: int | None) -> set[tuple[Object, Object]]:
        """World-frame AABB contact check, works for tilted objects."""
        pcid = self.env.physics_client_id
        on_relations: set[tuple[Object, Object]] = set()
        lower_candidates = [self._table_obj] + self._movable_objs
        for obj1 in self._movable_objs:
            obj1_id = self._pybullet_ids[obj1]
            if obj1_id == held_id:
                continue
            obj1_bottom_z = float(p.getAABB(obj1_id, physicsClientId=pcid)[0][2])
            for obj2 in lower_candidates:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                obj2_top_z = float(p.getAABB(obj2_id, physicsClientId=pcid)[1][2])
                if abs(obj1_bottom_z - obj2_top_z) >= 0.005:
                    continue
                if check_body_collisions(
                    obj1_id, obj2_id, pcid, distance_threshold=0.002
                ):
                    on_relations.add((obj1, obj2))
        return on_relations

    def _get_goal_atoms(self) -> set[GroundAtom]:
        assert self._milk_obj is not None and self._cup_obj is not None
        return {
            GroundAtom(
                self._pour_predicates.pouring,
                [self._robot_obj, self._milk_obj, self._cup_obj],
            )
        }

    def _get_atoms_from_obs(self, obs: dict) -> set[GroundAtom]:
        atoms = super()._get_atoms_from_obs(obs)

        if self._cup_obj is not None:
            atoms.add(GroundAtom(self._pour_predicates.is_cup, [self._cup_obj]))

        if self._milk_obj is not None and self._cup_obj is not None:
            assert self.env.scene is not None
            held_idx = int(obs["held_object_idx"][0])
            held_id = None
            if held_idx >= 0:
                held_id = self.env.scene.object_ids[held_idx]

            milk_id = self._pybullet_ids[self._milk_obj]
            cup_id = self._pybullet_ids[self._cup_obj]

            if held_id == milk_id:
                milk_pose = get_pose(milk_id, self.env.physics_client_id)
                cup_pose = get_pose(cup_id, self.env.physics_client_id)
                cup_aabb = p.getAABB(cup_id, physicsClientId=self.env.physics_client_id)
                cup_top_z = float(cup_aabb[1][2])
                milk_aabb = p.getAABB(
                    milk_id, physicsClientId=self.env.physics_client_id
                )
                milk_half_y = float(milk_aabb[1][1] - milk_aabb[0][1]) / 2
                milk_pos = np.array(milk_pose.position)
                cup_pos = np.array(cup_pose.position)
                ee_pose = self.env.robot.get_end_effector_pose()
                ee_z_world = p.rotateVector(ee_pose.orientation, [0, 0, 1])
                is_tilted = abs(float(ee_z_world[2])) < 0.9
                if (
                    float(np.linalg.norm(milk_pos[:2] - cup_pos[:2]))
                    < milk_half_y + 0.05
                    and milk_pos[2] > cup_top_z
                    and is_tilted
                ):
                    atoms.add(
                        GroundAtom(
                            self._pour_predicates.pouring,
                            [self._robot_obj, self._milk_obj, self._cup_obj],
                        )
                    )

        return atoms


POUR_PARAMS_SPACE = Box(
    low=np.array([], dtype=np.float32),
    high=np.array([], dtype=np.float32),
    shape=(0,),
)


class PourGroundController(GroundParameterizedController[dict, np.ndarray]):
    """Controller for positioning held milk carton above the cup for pouring.

    Moves to a lifting pose, then to above the cup (with slight -x
    offset), then tilts the carton 30 degrees around the x axis to
    simulate pouring.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        env: object,
        abstractor: TabletopPourAbstractor,
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
            raise TrajectorySamplingFailure("Failed to compute pour plan")

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
        return np.hstack([joint_delta, [0.0]]).astype(np.float32)

    def observe(self, x: dict) -> None:
        pass

    def _compute_kinematic_plan(self) -> list[KinematicState] | None:
        assert self.env.robot is not None
        assert self.env.scene is not None
        assert self._current_obs is not None

        _, milk_obj, cup_obj = self.objects
        milk_id = self.abstractor.get_pybullet_id(milk_obj)
        cup_id = self.abstractor.get_pybullet_id(cup_obj)
        table_id = self.env.scene.table_id

        robot_joints = list(self._current_obs["joint_positions"])
        if len(robot_joints) >= 9:
            finger_avg = (robot_joints[7] + robot_joints[8]) / 2
            robot_joints[7] = finger_avg
            robot_joints[8] = finger_avg

        object_poses: dict[int, Pose] = {
            table_id: Pose(position=(0.5, 0.0, -0.015)),
        }
        obs_poses = self._current_obs["object_poses"]
        for i, obj_id in enumerate(self.env.scene.object_ids):
            pos = tuple(obs_poses[i, :3])
            orn = tuple(obs_poses[i, 3:])
            object_poses[obj_id] = Pose(position=pos, orientation=orn)

        held_idx = int(self._current_obs["held_object_idx"][0])
        attachments: dict[int, Pose] = {}
        held_id = None
        grasp_tf = self._current_obs["grasp_transform"]
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        initial_state = KinematicState(robot_joints, object_poses, attachments)
        initial_state.set_pybullet(self.env.robot)
        state = initial_state
        plan = [state]

        cup_pose = get_pose(cup_id, self.env.physics_client_id)
        cup_aabb = p.getAABB(cup_id, physicsClientId=self.env.physics_client_id)
        cup_top_z = float(cup_aabb[1][2])

        milk_aabb = p.getAABB(milk_id, physicsClientId=self.env.physics_client_id)
        milk_height = float(milk_aabb[1][2] - milk_aabb[0][2])
        milk_y = float(milk_aabb[1][1] - milk_aabb[0][1]) / 2

        curr_ee_pose = self.env.robot.get_end_effector_pose()
        lifting_height = float(milk_aabb[1][2]) + milk_height
        pour_height = milk_height + 0.02

        pour_position = (
            float(cup_pose.position[0]),
            float(cup_pose.position[1]) - milk_y,
            cup_top_z + pour_height,
        )

        position_waypoints = [
            Pose(
                (curr_ee_pose.position[0], curr_ee_pose.position[1], lifting_height),
                curr_ee_pose.orientation,
            ),
            Pose(pour_position, curr_ee_pose.orientation),
        ]

        milk_attachment = attachments[milk_id]
        collision_ids = set(object_poses.keys()) - {milk_id, cup_id, table_id}

        for waypoint in position_waypoints:
            state.set_pybullet(self.env.robot)
            motion_plan = run_smooth_motion_planning_to_pose_with_surface_check(
                waypoint,
                self.env.robot,
                collision_ids=collision_ids,
                surface_id=table_id,
                end_effector_frame_to_plan_frame=Pose.identity(),
                seed=0,
                max_time=2.0,
                held_object=milk_id,
                base_link_to_held_obj=milk_attachment,
            )
            if motion_plan is None:
                return None
            for robot_joints_step in motion_plan:
                state = state.copy_with(robot_joints=robot_joints_step)
                plan.append(state)

        # Tilt
        state.set_pybullet(self.env.robot)
        pre_tilt_ee = self.env.robot.get_end_effector_pose()
        tilt_quat = p.getQuaternionFromEuler([np.pi / 6, 0, 0])
        post_tilt_pose = multiply_poses(pre_tilt_ee, Pose((0, 0, 0), tilt_quat))
        tilt_path = list(
            iter_between_poses(
                pre_tilt_ee, post_tilt_pose, num_interp=10, include_start=False
            )
        )
        joint_distance_fn = create_joint_distance_fn(self.env.robot)
        tilt_joints = smoothly_follow_end_effector_path(
            self.env.robot,
            tilt_path,
            initial_joints=state.robot_joints,
            collision_ids=collision_ids,
            joint_distance_fn=joint_distance_fn,
            held_object=milk_id,
            base_link_to_held_obj=milk_attachment,
            max_time=2.0,
            include_start=False,
        )
        for robot_joints_step in tilt_joints:
            state = state.copy_with(robot_joints=robot_joints_step)
            plan.append(state)

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


def create_tabletop_pour_operators(
    types: TabletopTypes, predicates: TabletopPourPredicates
) -> set[LiftedOperator]:
    """Create LiftedOperators for the pour task."""
    pick_op = create_pick_operator(types, predicates)

    robot = Variable("?robot", types.robot)
    obj = Variable("?obj", types.obj)
    cup = Variable("?cup", types.obj)

    pour_op = LiftedOperator(
        "pour",
        [robot, obj, cup],
        preconditions={
            LiftedAtom(predicates.holding, [robot, obj]),
            LiftedAtom(predicates.is_cup, [cup]),
        },
        add_effects={
            LiftedAtom(predicates.pouring, [robot, obj, cup]),
        },
        delete_effects=set(),
    )

    return {pick_op, pour_op}


def create_tabletop_pour_skills(
    types: TabletopTypes,
    operators: set[LiftedOperator],
    env: object,
    abstractor: TabletopPourAbstractor,
) -> set[LiftedSkill]:
    """Create LiftedSkills for the pour task."""
    assert isinstance(env, TabletopBaseEnv)

    robot_var = Variable("?robot", types.robot)
    obj_var = Variable("?obj", types.obj)
    cup_var = Variable("?cup", types.obj)
    surface_var = Variable("?surface", types.obj)

    pick_operator = next(op for op in operators if op.name == "pick")
    pour_operator = next(op for op in operators if op.name == "pour")

    def _obj_pick_factory(e: TabletopBaseEnv, a: TabletopPourAbstractor) -> Any:
        return lambda objects: ObjPickGroundController(objects, e, a)

    pick_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, obj_var, surface_var],
        controller_cls=_obj_pick_factory(env, abstractor),
        params_space=GRASP_PARAMS_SPACE,
    )
    pick_skill = LiftedSkill(operator=pick_operator, controller=pick_lifted)

    def _pour_factory(e: TabletopBaseEnv, a: TabletopPourAbstractor) -> Any:
        return lambda objects: PourGroundController(objects, e, a)

    pour_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, obj_var, cup_var],
        controller_cls=_pour_factory(env, abstractor),
        params_space=POUR_PARAMS_SPACE,
    )
    pour_skill = LiftedSkill(operator=pour_operator, controller=pour_lifted)

    return {pick_skill, pour_skill}
