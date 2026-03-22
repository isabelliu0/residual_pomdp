"""TAMP components for TabletopViewOcclusionEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
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
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    Predicate,
    Variable,
)

from residual_controllers.envs.tabletop_manipulation import (
    get_kinematic_plan_to_place_object,
)
from residual_controllers.envs.tabletop_tamp_base import (
    TabletopBaseAbstractor,
    TabletopBasePredicates,
    TabletopTypes,
    create_pick_operator,
    create_pick_skill,
)
from residual_controllers.envs.tabletop_view_occlusion import (
    TabletopViewOcclusionScene,
)
from residual_controllers.geometry import get_half_extents_from_aabb

if TYPE_CHECKING:
    from residual_controllers.envs.tabletop_base import TabletopBaseEnv


@dataclass
class TabletopPredicates(TabletopBasePredicates):
    """Predicates for TabletopViewOcclusionEnv."""

    def __init__(self, types: TabletopTypes) -> None:
        super().__init__(types)
        self.is_target = Predicate("is-target", [types.obj])
        self.is_target_area = Predicate("is-target-area", [types.obj])
        self.in_target_area = Predicate("in-target-area", [types.obj, types.obj])

    def as_set(self) -> set[Predicate]:
        return super().as_set() | {
            self.is_target,
            self.is_target_area,
            self.in_target_area,
        }


class TabletopAbstractor(TabletopBaseAbstractor):
    """Abstractor for TabletopViewOcclusionEnv."""

    def __init__(
        self,
        env: TabletopBaseEnv,
        types: TabletopTypes,
        predicates: TabletopPredicates,
    ) -> None:
        super().__init__(env, types, predicates)
        self._vo_predicates = predicates
        self._target_area_obj = Object("target_area", types.obj)
        self._target_id: int | None = None

    def _setup_extra_objects(self) -> None:
        scene = self.env.scene
        assert isinstance(scene, TabletopViewOcclusionScene)
        self._pybullet_ids[self._target_area_obj] = scene.target_area_id
        self._target_id = scene.object_ids[scene.target_idx]

    def _get_objects(self) -> set[Object]:
        return super()._get_objects() | {self._target_area_obj}

    def _get_goal_atoms(self) -> set[GroundAtom]:
        pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
        assert self._target_id is not None
        target_obj = pybullet_id_to_obj[self._target_id]
        return {
            GroundAtom(
                self._vo_predicates.in_target_area, [target_obj, self._target_area_obj]
            )
        }

    def _get_on_relations(self, held_id: int | None) -> set[tuple[Object, Object]]:
        pcid = self.env.physics_client_id
        upper_candidates = [self._target_area_obj] + self._movable_objs
        lower_candidates = [self._target_area_obj, self._table_obj] + self._movable_objs

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

                if abs(obj1_bottom_z - obj2_top_z) >= 0.005:
                    continue

                if obj2 == self._target_area_obj and obj1 in self._movable_objs:
                    block_x, block_y = pose1.position[0], pose1.position[1]
                    ta_x, ta_y = pose2.position[0], pose2.position[1]
                    if (
                        block_x - half1[0] >= ta_x - half2[0]
                        and block_x + half1[0] <= ta_x + half2[0]
                        and block_y - half1[1] >= ta_y - half2[1]
                        and block_y + half1[1] <= ta_y + half2[1]
                    ):
                        on_relations.add((obj1, obj2))
                        on_target_area.add(obj1)
                elif check_body_collisions(
                    obj1_id, obj2_id, pcid, distance_threshold=0.002
                ):
                    on_relations.add((obj1, obj2))

        return on_relations

    def _atoms_from_on_relation(self, obj1: Object, obj2: Object) -> set[GroundAtom]:
        if obj2 == self._target_area_obj and obj1 in self._movable_objs:
            return {GroundAtom(self._vo_predicates.in_target_area, [obj1, obj2])}
        return {GroundAtom(self.predicates.on, [obj1, obj2])}

    def _get_atoms_from_obs(self, obs: dict) -> set[GroundAtom]:
        atoms = super()._get_atoms_from_obs(obs)
        atoms.add(
            GroundAtom(self._vo_predicates.is_target_area, [self._target_area_obj])
        )
        for obj in self._movable_objs:
            if self._pybullet_ids[obj] == self._target_id:
                atoms.add(GroundAtom(self._vo_predicates.is_target, [obj]))
        return atoms


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
        env: TabletopBaseEnv,
        abstractor: TabletopAbstractor,
    ):
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

        return np.hstack([joint_delta, [gripper_action]]).astype(np.float32)

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
            self.env.scene.table_id: Pose(position=(0.5, 0.0, -0.015)),  # type: ignore[union-attr] # pylint: disable=line-too-long
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

        place_z = 0.025
        world_obj_pose = Pose(
            position=(area_x, area_y, place_z), orientation=(0, 0, 0, 1)
        )

        table_pose = get_pose(self.env.scene.table_id, self.env.physics_client_id)  # type: ignore[union-attr]  # pylint: disable=line-too-long
        relative_placement = multiply_poses(table_pose.invert(), world_obj_pose)

        def placement_gen():
            yield relative_placement

        collision_ids = set(object_poses.keys())
        plan = get_kinematic_plan_to_place_object(
            initial_state,
            self.env.robot,
            object_id,
            self.env.scene.table_id,  # type: ignore[union-attr]
            collision_ids,
            placement_generator=placement_gen(),  # type: ignore[no-untyped-call]
            placement_generator_iters=1,
            max_motion_planning_time=1.0,
            max_smoothing_iters_per_step=1,
            retract_after=False,
        )

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


def create_tabletop_operators(
    types: TabletopTypes, predicates: TabletopPredicates
) -> set[LiftedOperator]:
    """Create LiftedOperators for TabletopViewOcclusionEnv."""
    pick_op = create_pick_operator(types, predicates)

    robot = Variable("?robot", types.robot)
    obj = Variable("?obj", types.obj)
    target_area = Variable("?target_area", types.obj)

    place_op = LiftedOperator(
        "place",
        [robot, obj, target_area],
        preconditions={
            LiftedAtom(predicates.holding, [robot, obj]),
            LiftedAtom(predicates.is_target_area, [target_area]),
        },
        add_effects={
            LiftedAtom(predicates.in_target_area, [obj, target_area]),
            LiftedAtom(predicates.gripper_empty, [robot]),
        },
        delete_effects={
            LiftedAtom(predicates.holding, [robot, obj]),
        },
    )

    return {pick_op, place_op}


def create_tabletop_skills(
    types: TabletopTypes,
    operators: set[LiftedOperator],
    env: TabletopBaseEnv,
    abstractor: TabletopAbstractor,
) -> set[LiftedSkill]:
    """Create LiftedSkills for TabletopViewOcclusionEnv."""
    robot_var = Variable("?robot", types.robot)
    obj_var = Variable("?obj", types.obj)
    target_area_var = Variable("?target_area", types.obj)

    pick_operator = next(op for op in operators if op.name == "pick")
    place_operator = next(op for op in operators if op.name == "place")

    pick_skill = create_pick_skill(types, pick_operator, env, abstractor)

    def _place_factory(e: TabletopBaseEnv, a: TabletopAbstractor) -> Any:
        return lambda objects: PlaceGroundController(objects, e, a)

    place_lifted: LiftedParameterizedController = LiftedParameterizedController(
        variables=[robot_var, obj_var, target_area_var],
        controller_cls=_place_factory(env, abstractor),
        params_space=PLACE_PARAMS_SPACE,
    )
    place_skill = LiftedSkill(operator=place_operator, controller=place_lifted)

    return {pick_skill, place_skill}
