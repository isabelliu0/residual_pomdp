"""SeSamE TAMP runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import Plan, PlanningProblem, RelationalAbstractState
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import RelationalAbstractSuccessorGenerator
from gymnasium.spaces import Box
from relational_structs import GroundOperator, LiftedOperator

if TYPE_CHECKING:
    from bilevel_planning.structs import LiftedSkill

    from residual_controllers.tamp.structs import PlanningComponents


def create_sesame_planner(
    components: PlanningComponents,
    skills: set[LiftedSkill],
    state_abstractor: Callable,
    transition_function: Callable,
    heuristic_name: str = "hff",
    max_abstract_plans: int = 10,
    num_sampling_attempts_per_step: int = 5,
    max_trajectory_steps: int = 100,
    seed: int = 0,
):
    """Create a SesamePlanner for the given planning components."""
    successor_generator = RelationalAbstractSuccessorGenerator(components.operators)

    def abstract_successor_function(s: RelationalAbstractState):
        return successor_generator(s)

    abstract_plan_generator: RelationalHeuristicSearchAbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            types=components.types,
            predicates=components.predicates.as_set(),
            operators=components.operators,
            heuristic_name=heuristic_name,
            seed=seed,
        )
    )

    skills_by_operator = {skill.operator: skill for skill in skills}

    def controller_generator(abstract_action: GroundOperator):
        lifted_op = abstract_action.parent
        assert isinstance(lifted_op, LiftedOperator)
        if lifted_op not in skills_by_operator:
            raise ValueError(f"No skill found for operator {lifted_op.name}")
        skill = skills_by_operator[lifted_op]
        ground_skill = skill.ground(abstract_action.parameters)
        return ground_skill.controller

    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=controller_generator,
        transition_function=transition_function,
        state_abstractor=state_abstractor,
        max_trajectory_steps=max_trajectory_steps,
    )

    planner = SesamePlanner(
        abstract_plan_generator=abstract_plan_generator,
        trajectory_sampler=trajectory_sampler,
        max_abstract_plans=max_abstract_plans,
        num_sampling_attempts_per_step=num_sampling_attempts_per_step,
        abstract_successor_function=abstract_successor_function,
        state_abstractor=state_abstractor,
        seed=seed,
    )

    return planner


def run_tamp(
    components: PlanningComponents,
    skills: set[LiftedSkill],
    initial_state,
    goal,
    state_abstractor: Callable,
    transition_function: Callable,
    timeout: float = 30.0,
    seed: int = 0,
) -> tuple[Plan | None, Any]:
    """Run TAMP planning and return a plan and graph."""
    planner = create_sesame_planner(
        components=components,
        skills=skills,
        state_abstractor=state_abstractor,
        transition_function=transition_function,
        seed=seed,
    )

    # Create dummy spaces - they're not used by SesamePlanner for relational planning
    dummy_state_space: Any = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    dummy_action_space: Any = Box(
        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
    )

    problem = PlanningProblem(
        state_space=dummy_state_space,
        action_space=dummy_action_space,
        initial_state=initial_state,
        transition_function=transition_function,
        goal=goal,
    )

    plan, graph = planner.run(problem, timeout)
    return plan, graph
