"""Data collection for operating region estimation."""

from __future__ import annotations

import copy

from residual_controllers.beliefs.structs import Belief
from residual_controllers.operating_region.features import extract_features
from residual_controllers.operating_region.structs import SubsequenceRecord
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph


def _parse_operator_name(action_str: str) -> str:
    tokens = action_str.strip("() ").split()
    return tokens[0] if tokens else "unknown"


def _execute_plan(system, plan) -> bool:
    if plan is None:
        return False
    terminated = False
    for action in plan.actions:
        _, _, terminated, _, _ = system.env.step(action)
        if terminated:
            break
    return terminated


def _select_best_viewpoint(system, target_labels: list[str], seed: int | None):
    """Select the viewpoint with highest expected IG across all target
    objects."""
    best_vp = None
    best_ig = -1
    for label in target_labels:
        vp = system.select_viewpoint(label, seed=seed)
        if vp is not None:
            ig = system.get_viewpoint_info_gain(vp)
            if ig > best_ig:
                best_ig = ig
                best_vp = vp
    return best_vp


def _do_nbv_steps(
    system, n_steps: int, target_labels: list[str], seed: int | None
) -> int:
    """Execute up to n_steps of NBV in system.env, updating belief in-place.

    At each viewpoint selection, picks the target object with highest
    expected IG. Returns the number of steps actually executed.
    """
    if n_steps <= 0:
        return 0

    steps_done = 0
    viewpoint = _select_best_viewpoint(system, target_labels, seed)
    if viewpoint is None:
        return steps_done

    trajectory = system.plan_to_viewpoint(viewpoint)
    traj_idx = 0

    while steps_done < n_steps:
        if traj_idx >= len(trajectory) - 1:
            viewpoint = _select_best_viewpoint(system, target_labels, seed)
            if viewpoint is None:
                break
            trajectory = system.plan_to_viewpoint(viewpoint)
            traj_idx = 0

        action = system.action_from_trajectory(
            trajectory[traj_idx], trajectory[traj_idx + 1]
        )
        system.env.step(action)
        traj_idx += 1
        steps_done += 1

    return steps_done


def collect_episode(
    system,
    seed: int,
    nbv_step_counts: tuple[int, ...] = (0, 10, 20, 40),
) -> list[SubsequenceRecord]:
    """Collect SubsequenceRecords for one episode.

    We use information gathering to augment the distribution of belief
    states we see for each subsequence, which is important for training
    a robust operating region classifier.
    """
    system.reset(seed=seed)

    symbolic_plan = system.get_symbolic_plan()
    if symbolic_plan is None:
        return []

    ignored = system.get_oig_ignored_objects()
    subsequences = build_object_interaction_graph(symbolic_plan, ignored)

    records: list[SubsequenceRecord] = []

    for subseq_actions, obj_names in subsequences:
        all_obj_labels = system.env.object_labels
        relevant_labels = list(system.get_object_labels_for_names(obj_names).values())
        goal_add, _ = system.get_subsequence_effects(subseq_actions)

        operator_name = "+".join(_parse_operator_name(a) for a in subseq_actions)

        pre_state = system.env.get_state()
        pre_belief: Belief = copy.deepcopy(system.env.belief)

        if relevant_labels and pre_belief is not None:
            for n_nbv in nbv_step_counts:
                print(f"===== NBV steps: {n_nbv} =====")
                system.env.set_state(pre_state)
                system.env.belief = copy.deepcopy(pre_belief)

                steps_done = _do_nbv_steps(system, n_nbv, relevant_labels, seed)

                feats = extract_features(
                    system.env.belief, all_obj_labels, relevant_labels
                )

                plan = system.plan_for_goal(goal_add, seed=seed)
                print("Planned! Ready to execute plan for goal_add:", goal_add)
                success = _execute_plan(system, plan)
                print(f"Executed! Success: {success}")

                records.append(
                    SubsequenceRecord(
                        operator_name=operator_name,
                        relevant_object_labels=relevant_labels,
                        features=feats,
                        success=success,
                        source="nbv",
                        metadata={
                            "nbv_steps_requested": n_nbv,
                            "nbv_steps_done": steps_done,
                        },
                    )
                )

    return records
