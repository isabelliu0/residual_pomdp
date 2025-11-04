"""Action execution with residual policy correction."""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
from tampura.structs import Action, AliasStore
from tampura.symbolic import Atom, Not, OneOf
from tampura_environments.slam_collect.env import generate_trajectory

from residual_controllers.planner import get_base_action_name


def execute_action_sequence(
    symbolic_action: Action,
    belief: Any,
    store: AliasStore,
    spec: Any,
    base_controller: Callable[[Action, Any, AliasStore], list],
    residual_policy: Any | None = None,
    target_skill: str | None = None,
    belief_encoder: Callable[[Any], np.ndarray] | None = None,
) -> tuple[Any, list[int], np.ndarray | None]:
    """Execute a symbolic action by getting primitives and stepping through
    simulation.

    This function:
    1. Gets primitive action sequence from base controller (e.g., RRT waypoints)
    2. For each primitive action:
       - If skill matches target_skill and residual_policy exists, add residual
       - Step through simulation to get next belief
    3. Returns final belief, verified effects, and residual action used
    """
    action_base_name = get_base_action_name(symbolic_action.name)

    # Get primitive action sequence from base controller
    primitive_actions = base_controller(symbolic_action, belief, store)

    if primitive_actions is None or len(primitive_actions) == 0:
        # Controller failed, return current belief with no effects satisfied
        return belief, [], None

    # Check if we should apply residual policy
    apply_residual = (
        residual_policy is not None
        and target_skill is not None
        and action_base_name == target_skill
    )

    residual_action = None
    if apply_residual and belief_encoder is not None:
        # Apply residual to each primitive action
        modified_primitives = []
        encoded_belief = belief_encoder(belief)

        # Get residual action (use exploration noise during training)
        assert residual_policy is not None
        residual_action = residual_policy.predict(encoded_belief, deterministic=False)

        for primitive in primitive_actions:
            modified_primitive = _add_residual_to_primitive(primitive, residual_action)
            modified_primitives.append(modified_primitive)
        primitive_actions = modified_primitives

    # Execute primitives on all particles
    trajectories, _ = generate_trajectory(belief.particles, primitive_actions, std=0.04)
    next_belief = _create_belief_from_trajectories(belief, trajectories)
    action_schema = spec.get_action_schema(action_base_name)
    satisfied_veffects = _check_verified_effects(next_belief, action_schema, store)

    return next_belief, satisfied_veffects, residual_action


def _add_residual_to_primitive(primitive: Any, residual: np.ndarray) -> Any:
    """Add residual correction to a primitive action."""
    modified = copy.deepcopy(primitive)

    # For SLAM2D, primitives are SlamAction(dx, dy, attach, detach)
    if hasattr(modified, "dx") and hasattr(modified, "dy"):
        if len(residual) >= 2:
            modified.dx += float(residual[0])
            modified.dy += float(residual[1])

    return modified


def _create_belief_from_trajectories(
    current_belief: Any, trajectories: list[list[Any]]
) -> Any:
    """Create next belief from simulation trajectories."""
    next_belief = copy.deepcopy(current_belief)
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) > 0:
            next_belief.particles[i] = trajectory[-1]
    return next_belief


def _check_verified_effects(
    belief: Any, action_schema: Any, store: AliasStore
) -> list[int]:
    """Check which verified effects are satisfied in the resulting belief."""
    if (
        not hasattr(action_schema, "verify_effects")
        or len(action_schema.verify_effects) == 0
    ):
        raise ValueError("Action schema has no verified effects to check.")

    abstract_belief = belief.abstract(store)
    abstract_items = set(abstract_belief.items)

    satisfied = []
    for verify_effect in action_schema.verify_effects:
        satisfied.append(_check_single_effect(verify_effect, abstract_items))

    return satisfied


def _check_single_effect(effect: Any, abstract_items: set) -> int:
    """Check if a single verify_effect is satisfied."""
    if isinstance(effect, Atom):
        return 1 if effect in abstract_items else 0
    if isinstance(effect, Not):
        return 1 if effect.component not in abstract_items else 0
    if isinstance(effect, OneOf):
        # For OneOf, return the index of the satisfied component
        for idx, component in enumerate(effect.components):
            if _check_single_effect(component, abstract_items) == 1:
                return idx
        return -1  # None satisfied
    return 0
