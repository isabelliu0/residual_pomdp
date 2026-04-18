"""TAMP components for TabletopViewOcclusionEnv."""

from __future__ import annotations

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
    _make_checking_state,
    create_pick_operator,
    create_pick_skill,
)
from residual_controllers.envs.tabletop_view_occlusion import (
    TabletopViewOcclusionScene,
)

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
            aabb1 = p.getAABB(obj1_id, physicsClientId=pcid)
            obj1_bottom_z = float(aabb1[0][2])

            for obj2 in lower_candidates:
                if obj1 == obj2:
                    continue
                if obj2 == self._table_obj and obj1 in on_target_area:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                aabb2 = p.getAABB(obj2_id, physicsClientId=pcid)
                obj2_top_z = float(aabb2[1][2])

                if abs(obj1_bottom_z - obj2_top_z) >= 0.005:
                    continue

                if obj2 == self._target_area_obj and obj1 in self._movable_objs:
                    if (
                        aabb1[0][0] >= aabb2[0][0]
                        and aabb1[1][0] <= aabb2[1][0]
                        and aabb1[0][1] >= aabb2[0][1]
                        and aabb1[1][1] <= aabb2[1][1]
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
    """Controller for placing a held object with PrePlace / Place / PostPlace
    phases."""

    def __init__(
        self,
        objects: Sequence[Object],
        env: TabletopBaseEnv,
        abstractor: TabletopAbstractor,
    ):
        super().__init__(objects)
        self.env = env
        self.abstractor = abstractor
        self._plan: Sequence[KinematicState | SimpleNamespace] | None = None
        self._detach_idx: int = -1
        self._action_idx = 0
        self._phase = "pre_place"
        self._terminated = False
        self._current_obs: dict | None = None

    def sample_parameters(self, x: dict, _rng: np.random.Generator) -> dict[str, float]:
        return {}

    def reset(self, x: dict, params: dict[str, float]) -> None:
        self._terminated = False
        self._action_idx = 0
        self._phase = "pre_place"
        self._current_obs = x
        self._plan = self._compute_kinematic_plan()
        if self._plan is None:
            raise TrajectorySamplingFailure("Failed to compute place plan")
        self._detach_idx = next(
            (
                i
                for i in range(1, len(self._plan))
                if self._plan[i - 1].attachments and not self._plan[i].attachments
            ),
            -1,
        )
        if self._detach_idx < 0:
            raise TrajectorySamplingFailure("No release transition found in place plan")

    def reset_for_checking(self, plan: Any, initial_obs: dict) -> None:
        """Reset for checking with a given plan and initial observation."""
        self._terminated = False
        self._action_idx = 0
        self._phase = "pre_place"
        self._current_obs = initial_obs
        states = list(plan.states)
        self._detach_idx = next(
            (
                i
                for i in range(1, len(states))
                if int(states[i]["held_object_idx"][0])
                < 0
                <= int(states[i - 1]["held_object_idx"][0])
            ),
            -1,
        )
        if self._detach_idx < 0:
            raise TrajectorySamplingFailure(
                "No release transition found in plan states"
            )
        self._plan = [_make_checking_state(s) for s in states]

    def terminated(self) -> bool:
        return self._terminated

    def observe(self, x: dict) -> None:
        self._current_obs = x

    def step(self) -> np.ndarray:
        if self._phase == "place":
            assert self._current_obs is not None
            if int(self._current_obs["held_object_idx"][0]) >= 0:
                raise TrajectorySamplingFailure("Place failed: object not released")
            self._phase = "post_place"

        if self._plan is None or self._action_idx >= len(self._plan) - 1:
            self._terminated = True
            return np.zeros(8, dtype=np.float32)

        s0 = self._plan[self._action_idx]
        s1 = self._plan[self._action_idx + 1]
        self._action_idx += 1

        if self._action_idx == self._detach_idx:
            assert self._current_obs is not None
            ee_xy = np.array(self._current_obs["camera_pose"][:2], dtype=float)
            target_xy = np.array(self._current_obs["target_area_pose"][:2], dtype=float)
            if np.linalg.norm(ee_xy - target_xy) > 0.05:
                raise TrajectorySamplingFailure("PrePlace failed: EE not above target")
            self._phase = "place"

        if self._action_idx >= len(self._plan) - 1:
            self._terminated = True

        joint_delta = np.subtract(s1.robot_joints[:7], s0.robot_joints[:7])
        gripper_action = 1.0 if (s0.attachments and not s1.attachments) else 0.0

        return np.hstack([joint_delta, [gripper_action]]).astype(np.float32)

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
            object_poses[obj_id] = Pose(
                position=tuple(obs_poses[i, :3]),
                orientation=tuple(obs_poses[i, 3:]),
            )

        held_idx = int(self._current_obs["held_object_idx"][0])
        grasp_tf = self._current_obs["grasp_transform"]
        attachments: dict[int, Pose] = {}
        held_id = None
        if held_idx >= 0:
            held_id = self.env.scene.object_ids[held_idx]
            attachments[held_id] = Pose(
                position=tuple(grasp_tf[:3]), orientation=tuple(grasp_tf[3:])
            )

        initial_state = KinematicState(robot_joints, object_poses, attachments)
        initial_state.set_pybullet(self.env.robot)

        place_z = 0.025
        world_obj_pose = Pose(
            position=(area_x, area_y, place_z), orientation=(0, 0, 0, 1)
        )
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
            retract_after=True,
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
