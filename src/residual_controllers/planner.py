"""SymK planner interface."""

from __future__ import annotations

import logging
import tempfile
from typing import Any

from tampura.solvers.symk import symk_search, symk_translate
from tampura.spec import ProblemSpec
from tampura.structs import Action, AliasStore
from tampura.symbolic import ACTION_EXT, VEFFECT_SEPARATOR


def get_plans(
    spec: ProblemSpec,
    abstract_belief: Any,
    store: AliasStore,
    num_plans: int = 10,
    symk_config: dict[str, Any] | None = None,
) -> list[list[Action]]:
    """Get top-K plans from SymK with verified effects encoding."""
    if symk_config is None:
        symk_config = {
            "symk_direction": "bd",
            "symk_simple": True,
            "symk_selection": "top_k",
            "num_skeletons": num_plans,
        }
    else:
        symk_config["num_skeletons"] = num_plans

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_file, problem_file = spec.save_pddl(
                abstract_belief=abstract_belief,
                default_cost=100,
                folder=tmpdir,
                store=store,
            )

            sas_file = symk_translate(domain_file, problem_file)
            plans = symk_search(sas_file, symk_config)

            return plans if plans else []

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(f"SymK planning failed: {e}")
        return []


def filter_plans_with_skill(
    plans: list[list[Action]], skill_name: str
) -> list[list[Action]]:
    """Filter plans that contain the target skill."""
    filtered = []
    for plan in plans:
        for action in plan:
            action_base_name = (
                action.name.split(ACTION_EXT)[0]
                if ACTION_EXT in action.name
                else action.name
            )
            if action_base_name == skill_name:
                filtered.append(plan)
                break
    return filtered


def parse_verified_effects(action_name: str) -> list[int] | None:
    """Parse verified effects from SymK action encoding."""
    if ACTION_EXT not in action_name:
        return None

    _, veffect_str = action_name.split(ACTION_EXT)
    if len(veffect_str) == 0:
        return None

    return [int(v) for v in veffect_str.split(VEFFECT_SEPARATOR)]


def get_base_action_name(action_name: str) -> str:
    """Extract base action name without verified effects encoding."""
    return (
        action_name.split(ACTION_EXT)[0] if ACTION_EXT in action_name else action_name
    )
