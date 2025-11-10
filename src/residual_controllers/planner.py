"""SymK planner interface."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """PDDL action with name and arguments."""

    name: str
    args: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, tuple(self.args)))

    def __eq__(self, other):
        return self.name == other.name and self.args == other.args


def parse_action_from_line(line: str) -> Action | None:
    """Parse action from PDDL line."""
    match = re.match(r"\(([^ ]+)(?: (.*?))?\s*\)", line)
    if match:
        action_name = match.group(1)
        action_args = match.group(2).split() if match.group(2) else []
        return Action(name=action_name, args=action_args)
    return None


def parse_actions_from_file(filename: str) -> list[Action]:
    """Parse actions from plan file."""
    actions = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            action = parse_action_from_line(line.strip())
            if action:
                actions.append(action)
    return actions


def symk_translate(domain_file: str, problem_file: str, symk_path: str) -> str:
    """Translate PDDL to SAS format."""
    domain_dir = os.path.dirname(domain_file)
    sas_file = os.path.join(domain_dir, "output.sas")

    cmd = [
        "python",
        symk_path,
        "--sas-file",
        sas_file,
        "--translate",
        domain_file,
        problem_file,
        "--translate-options",
        "--keep-unimportant-variables",
        "--keep-unreachable-facts",
    ]

    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return sas_file


def symk_search(
    sas_file: str, config: dict[str, Any], symk_path: str
) -> list[list[Action]]:
    """Run SymK search."""
    domain_dir = os.path.dirname(sas_file)

    cmd = [
        "python",
        symk_path,
        "--search-time-limit",
        "20",
        "--plan-file",
        os.path.join(domain_dir, "sas_plan"),
        sas_file,
        "--search-options",
        "--search",
        f"symk-{config['symk_direction']}("
        f"simple={str(config['symk_simple']).lower()},"
        f"plan_selection={config['symk_selection']}("
        f"num_plans={config['num_skeletons']},dump_plans=false))",
    ]

    subprocess.run(cmd, capture_output=True, text=True, check=False)

    plan_files = [
        os.path.join(domain_dir, f)
        for f in os.listdir(domain_dir)
        if f.startswith("sas_plan")
    ]

    if not plan_files:
        return []

    return [parse_actions_from_file(pf) for pf in plan_files]


def get_plans(
    domain_file: str,
    problem_file: str,
    num_plans: int = 10,
    symk_config: dict[str, Any] | None = None,
    symk_path: str | None = None,
) -> list[list[Action]]:
    """Get top-K plans from SymK."""
    if symk_config is None:
        symk_config = {
            "symk_direction": "bd",
            "symk_simple": True,
            "symk_selection": "top_k",
            "num_skeletons": num_plans,
        }
    else:
        symk_config["num_skeletons"] = num_plans

    if symk_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        symk_path = os.path.join(script_dir, "../../third_party/symk/fast-downward.py")
        if not os.path.exists(symk_path):
            raise FileNotFoundError(f"SymK not found at {symk_path}")

    try:
        sas_file = symk_translate(domain_file, problem_file, symk_path)
        return symk_search(sas_file, symk_config, symk_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(f"SymK planning failed: {e}")
        return []


def filter_plans_with_skill(
    plans: list[list[Action]], skill_name: str
) -> list[list[Action]]:
    """Filter plans containing target skill."""
    filtered = []
    for plan in plans:
        if any(action.name.startswith(skill_name) for action in plan):
            filtered.append(plan)
    return filtered


def get_base_action_name(action_name: str) -> str:
    """Extract base action name."""
    return action_name.split("_cm_")[0] if "_cm_" in action_name else action_name
