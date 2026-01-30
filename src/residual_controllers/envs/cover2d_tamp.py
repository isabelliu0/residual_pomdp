"""TAMP components for Cover2D environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractState,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

from residual_controllers.envs.cover2d import (
    Action,
    Belief,
    PickController,
    PlaceController,
    State,
    World,
    get_mean_state,
)
from residual_controllers.tamp.structs import (
    BeliefAbstractor,
    PredicateContainer,
    TypeContainer,
)


@dataclass
class Cover2DTypes(TypeContainer):
    """Container for Cover2D types."""

    def __init__(self) -> None:
        self.object = Type("object")

    def as_set(self) -> set[Type]:
        return {self.object}


@dataclass
class Cover2DPredicates(PredicateContainer):
    """Container for Cover2D predicates."""

    def __init__(self, types: Cover2DTypes) -> None:
        self.holding = Predicate("holding", [types.object])
        self.at_goal = Predicate("at-goal", [types.object])
        self.free_gripper = Predicate("free-gripper", [])

    def __getitem__(self, key: str) -> Predicate:
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        return {self.holding, self.at_goal, self.free_gripper}


class Cover2DAbstractor(BeliefAbstractor[Belief]):
    """Belief abstractor for Cover2D environment."""

    def __init__(
        self,
        world: World,
        types: Cover2DTypes,
        predicates: Cover2DPredicates,
    ):
        self.world = world
        self.types = types
        self.predicates = predicates
        self._objects: set[Object] = set()
        self._obj_map: dict[str, Object] = {}

    def reset(  # pylint: disable=arguments-renamed
        self, belief: Belief
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        state = get_mean_state(belief)
        self._objects = set()
        self._obj_map = {}

        for obj_id in state.object_poses:
            obj_name = f"obj{obj_id}"
            obj_obj = Object(obj_name, self.types.object)
            self._obj_map[obj_name] = obj_obj
            self._objects.add(obj_obj)

        init_atoms = self._get_atoms(state)

        goal_atoms = set()
        for obj_obj in self._objects:
            goal_atoms.add(GroundAtom(self.predicates.at_goal, [obj_obj]))

        return self._objects, init_atoms, goal_atoms

    def step(self, belief: Belief):  # pylint: disable=arguments-renamed
        state = get_mean_state(belief)
        atoms = self._get_atoms(state)
        return RelationalAbstractState(atoms=atoms, objects=self._objects)

    def _get_atoms(self, state: State) -> set[GroundAtom]:
        atoms = set()

        if state.gripper_state.is_holding:
            if state.gripper_state.held_object_id is not None:
                held_obj_name = f"obj{state.gripper_state.held_object_id}"
                if held_obj_name in self._obj_map:
                    atoms.add(
                        GroundAtom(
                            self.predicates.holding, [self._obj_map[held_obj_name]]
                        )
                    )
        else:
            atoms.add(GroundAtom(self.predicates.free_gripper, []))

        for obj_id, obj_pose in state.object_poses.items():
            obj_name = f"obj{obj_id}"
            if obj_name not in self._obj_map:
                continue
            if self.world.in_goal_region(obj_pose.x, obj_pose.y):
                atoms.add(
                    GroundAtom(self.predicates.at_goal, [self._obj_map[obj_name]])
                )

        return atoms


def create_cover2d_operators(
    types: Cover2DTypes, predicates: Cover2DPredicates
) -> set[LiftedOperator]:
    """Create lifted operators for Cover2D."""
    obj = Variable("?obj", types.object)

    pick_op = LiftedOperator(
        "pick",
        [obj],
        preconditions={
            LiftedAtom(predicates.free_gripper, []),
        },
        add_effects={
            LiftedAtom(predicates.holding, [obj]),
        },
        delete_effects={
            LiftedAtom(predicates.free_gripper, []),
        },
    )

    place_op = LiftedOperator(
        "place",
        [obj],
        preconditions={
            LiftedAtom(predicates.holding, [obj]),
        },
        add_effects={
            LiftedAtom(predicates.at_goal, [obj]),
            LiftedAtom(predicates.free_gripper, []),
        },
        delete_effects={
            LiftedAtom(predicates.holding, [obj]),
        },
    )

    return {pick_op, place_op}


class PickGroundController(GroundParameterizedController[Belief, Action]):
    """Ground controller for picking objects in Cover2D."""

    def __init__(self, objects: Sequence[Object], world: World):
        super().__init__(objects)
        self.world = world
        self._controller: PickController | None = None
        self._terminated = False
        self._current_belief: Belief | None = None

    def sample_parameters(self, x: Belief, rng: np.random.Generator) -> Any:
        return {}

    def reset(self, x: Belief, params: Any) -> None:
        assert len(self.objects) == 1
        obj_name = self.objects[0].name
        obj_id = int(obj_name.replace("obj", ""))
        self._controller = PickController(self.world, obj_id)
        self._terminated = False
        self._current_belief = x

    def terminated(self) -> bool:
        return self._terminated

    def step(self) -> Action:
        assert self._controller is not None
        assert self._current_belief is not None
        action = self._controller.get_action(self._current_belief)
        return action

    def observe(self, x: Belief) -> None:
        self._current_belief = x
        if x.particles[0].gripper_state.is_holding:
            obj_id = int(self.objects[0].name.replace("obj", ""))
            if x.particles[0].gripper_state.held_object_id == obj_id:
                self._terminated = True


class PlaceGroundController(GroundParameterizedController[Belief, Action]):
    """Ground controller for placing objects in Cover2D."""

    def __init__(self, objects: Sequence[Object], world: World):
        super().__init__(objects)
        self.world = world
        self._controller: PlaceController | None = None
        self._terminated = False
        self._current_belief: Belief | None = None

    def sample_parameters(self, x: Belief, rng: np.random.Generator) -> Any:
        return {}

    def reset(self, x: Belief, params: Any) -> None:
        target_x = self.world.goal_region.x + self.world.goal_region.width / 2
        target_y = self.world.goal_region.y + self.world.goal_region.height / 2
        self._controller = PlaceController(self.world, target_x, target_y)
        self._terminated = False
        self._current_belief = x

    def terminated(self) -> bool:
        return self._terminated

    def step(self) -> Action:
        assert self._controller is not None
        assert self._current_belief is not None
        action = self._controller.get_action(self._current_belief)
        return action

    def observe(self, x: Belief) -> None:
        self._current_belief = x
        obj_id = int(self.objects[0].name.replace("obj", ""))
        if obj_id in x.particles[0].object_poses:
            obj_pose = x.particles[0].object_poses[obj_id]
            if self.world.in_goal_region(obj_pose.x, obj_pose.y):
                self._terminated = True


def create_cover2d_skills(
    types: Cover2DTypes,
    _predicates: Cover2DPredicates,
    operators: set[LiftedOperator],
    world: World,
) -> set[LiftedSkill]:
    """Create lifted skills for Cover2D."""
    obj_var = Variable("?obj", types.object)

    pick_operator = next(op for op in operators if op.name == "pick")
    place_operator = next(op for op in operators if op.name == "place")

    def pick_controller_factory(world: World) -> Any:
        return lambda objects: PickGroundController(objects, world)

    def place_controller_factory(world: World) -> Any:
        return lambda objects: PlaceGroundController(objects, world)

    pick_lifted_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            variables=[obj_var],
            controller_cls=pick_controller_factory(world),
        )
    )

    place_lifted_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            variables=[obj_var],
            controller_cls=place_controller_factory(world),
        )
    )

    pick_skill = LiftedSkill(operator=pick_operator, controller=pick_lifted_controller)
    place_skill = LiftedSkill(
        operator=place_operator, controller=place_lifted_controller
    )

    return {pick_skill, place_skill}
