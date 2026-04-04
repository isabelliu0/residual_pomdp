"""Train per-operator compliant RL policies for any TabletopBaseSystem."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)

from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.benchmarks.tabletop_view_occlusion_system import (
    TabletopViewOcclusionTAMPSystem,
)
from residual_controllers.operating_region.data_collect import (
    _check_operator_success,
    _do_nbv_steps,
    _execute_plan,
    _parse_operator_name,
)
from residual_controllers.operating_region.features import extract_features
from residual_controllers.operating_region.predictor import OperatingRegionPredictor
from residual_controllers.rl import RLPolicy, RLTrainer, encode_belief_for_rl
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph

_SYSTEMS: dict[str, Any] = {
    "nut_assembly": NutAssemblyTAMPSystem,
    "tabletop_view_occlusion": TabletopViewOcclusionTAMPSystem,
}

_ENV_DATA_DIRS: dict[str, str] = {
    "nut_assembly": "data/nut_assembly",
    "tabletop_view_occlusion": "data/tabletop_view_occlusion",
}


def _execute_op_with_phase_checks(
    system: Any, plan: Any, action_str: str, op_goal: set
) -> tuple[bool, bool, int]:
    """Execute plan actions with phase checks, stopping when op_goal is
    achieved.

    Returns (failure_detected, env_terminated, actions_consumed).
    """
    ctrl = None
    get_ctrl = getattr(system, "get_grounded_controller", None)
    if get_ctrl is not None:
        ctrl = get_ctrl(action_str)

    obs = system.env.get_observation()
    if ctrl is not None:
        try:
            ctrl.reset_for_checking(plan, obs)
        except TrajectorySamplingFailure:
            return True, False, 0

    first = True
    consumed = 0
    for action in plan.actions:
        if ctrl is not None:
            if not first:
                ctrl.observe(system.env.get_observation())
            try:
                ctrl.step()
            except TrajectorySamplingFailure:
                return True, False, consumed
            first = False

        _, _, terminated, _, _ = system.env.step(action)
        consumed += 1
        if terminated:
            return False, True, consumed
        if _check_operator_success(system, op_goal):
            return False, False, consumed

    return False, False, consumed


def _run_rl_for_operator(
    system: Any,
    op_goal: set,
    rel_labels: list[str],
    sigma_thresh: float,
    policy: RLPolicy,
    trainer: RLTrainer,
    max_steps: int,
    seed: int,
    n_fold: dict[str, int],
) -> tuple[bool, bool]:
    """Run RL loop for one operator.

    Returns (succeeded, env_terminated).
    """
    all_labels = system.env.object_labels
    for _ in range(max_steps):
        obs = encode_belief_for_rl(system.env.belief, rel_labels, n_fold=n_fold)
        action7 = policy.predict(obs, deterministic=False)
        system.env.step(np.append(action7, 0.0).astype(np.float32))
        next_obs = encode_belief_for_rl(system.env.belief, rel_labels, n_fold=n_fold)

        feats = extract_features(system.env.belief, all_labels, rel_labels)
        success = feats.relevant_sigma <= sigma_thresh
        trainer.store_transition(
            obs, action7, 1.0 if success else 0.0, next_obs, done=success
        )
        if trainer.should_train():
            trainer.train_step()

        if success:
            try:
                plan2 = system.plan_for_goal(op_goal, seed=seed)
            except Exception:  # pylint: disable=broad-exception-caught
                return False, False
            env_terminated = _execute_plan(system, plan2)
            return True, env_terminated
    return False, False


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    system_cls = _SYSTEMS[args.system]
    env_dir = _ENV_DATA_DIRS[args.system]
    predictor_path = Path(
        args.operator_predictor or f"{env_dir}/operator_region_predictor.pkl"
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    op_predictor = OperatingRegionPredictor()
    op_predictor.load(predictor_path)
    print(f"Loaded operator predictor from {predictor_path}")
    print(f"Fitted operators: {op_predictor.fitted_operators}")

    sigma_thresholds: dict[str, float] = {
        op: op_predictor.find_sigma_threshold(op, threshold=args.sigma_threshold)
        for op in op_predictor.fitted_operators
    }
    print(f"Sigma thresholds: {sigma_thresholds}")

    policy_cfg = {
        "action_dim": 7,
        "backend": "td3",
        "learning_rate": 3e-4,
        "noise_std": 0.1,
        "gamma": 0.99,
        "buffer_size": 100000,
        "device": "cpu",
    }
    trainer_cfg = {"gradient_steps": 1, "train_freq": 1, "min_buffer_size": 256}

    policies: dict[str, RLPolicy] = {}
    trainers: dict[str, RLTrainer] = {}

    episode_successes = 0

    system = system_cls(seed=args.seed, gui=args.gui)
    system.env.reset(seed=args.seed)
    n_fold: dict[str, int] = {
        lbl: system.env.get_belief_config().get_n_fold(lbl)
        for lbl in system.env.object_labels
    }
    print(f"Per-label n_fold symmetry: {n_fold}")
    try:
        for ep in range(args.num_episodes):
            episode_seed = args.seed + ep
            print(
                f"\n=== Episode {ep + 1}/{args.num_episodes} (seed={episode_seed}) ==="
            )

            system.reset(seed=episode_seed)
            symbolic_plan = system.get_symbolic_plan()
            if symbolic_plan is None:
                print("  No symbolic plan found, skipping.")
                continue

            ignored = system.get_oig_ignored_objects()
            subsequences = build_object_interaction_graph(symbolic_plan, ignored)

            episode_success = False
            for subseq_actions, obj_names in subsequences:
                rel_labels = list(
                    system.get_object_labels_for_names(obj_names).values()
                )
                obs_dim = len(rel_labels) * 14 + 1

                n_nbv = int(np.random.choice([0, 10, 20, 40]))
                if n_nbv > 0 and rel_labels:
                    print(f"  Running {n_nbv} NBV steps...")
                    _do_nbv_steps(system, n_nbv, rel_labels, episode_seed)

                plan_actions: list | None = None
                plan_states: list | None = None
                action_idx = 0

                for action_str in subseq_actions:
                    op_name = _parse_operator_name(action_str)
                    op_goal, _ = system.get_subsequence_effects([action_str])

                    if op_name not in policies:
                        policies[op_name] = RLPolicy(
                            observation_dim=obs_dim, seed=ep, **policy_cfg
                        )
                        trainers[op_name] = RLTrainer(policies[op_name], **trainer_cfg)
                        print(
                            f"  Created policy for operator '{op_name}' (obs_dim={obs_dim})"  # pylint: disable=line-too-long
                        )

                    sigma_thresh = sigma_thresholds.get(op_name, 0.0)

                    if plan_actions is None:
                        plan = system.plan(seed=episode_seed)
                        if plan is None:
                            print(f"  Planning failed for '{op_name}', skipping.")
                            break
                        plan_actions = list(plan.actions)
                        plan_states = list(plan.states)
                        action_idx = 0

                    op_plan = SimpleNamespace(
                        actions=plan_actions[action_idx:],
                        states=plan_states[action_idx:],
                    )
                    phase_failure, env_terminated, consumed = (
                        _execute_op_with_phase_checks(
                            system, op_plan, action_str, op_goal
                        )
                    )
                    action_idx += consumed

                    if env_terminated:
                        episode_success = True
                        break

                    if phase_failure or not _check_operator_success(system, op_goal):
                        reason = "phase failure" if phase_failure else "plan failed"
                        print(
                            f"  '{op_name}' {reason} — activating RL "
                            f"(sigma_thresh={sigma_thresh:.6f})"
                        )
                        rl_ok, rl_terminated = _run_rl_for_operator(
                            system,
                            op_goal,
                            rel_labels,
                            sigma_thresh,
                            policies[op_name],
                            trainers[op_name],
                            args.max_rl_steps,
                            episode_seed,
                            n_fold,
                        )
                        if rl_terminated:
                            episode_success = True
                            break
                        if not rl_ok:
                            print(f"  RL timed out for '{op_name}'.")
                            break
                        print(f"  RL succeeded for '{op_name}'.")
                        plan_actions = None  # re-plan before next operator
                else:
                    continue
                break
            if episode_success:
                episode_successes += 1

            print(
                f"  Episode {'SUCCESS' if episode_success else 'FAILURE'} "
                f"({episode_successes}/{ep + 1})"
            )
            for op_name, trainer in trainers.items():
                stats = trainer.get_stats()
                print(
                    f"  [{op_name}] transitions={stats['num_transitions']}, "
                    f"updates={stats['num_updates']}, buffer={stats['buffer_size']}"
                )

            if (ep + 1) % args.save_freq == 0:
                for op_name, trainer in trainers.items():
                    trainer.save(str(save_dir / f"{op_name}_ep{ep + 1}"))
                print(f"  Checkpoints saved to {save_dir}")

    finally:
        system.close()

    print("\n=== Training complete ===")
    print(f"Success rate: {episode_successes}/{args.num_episodes}")
    for op_name, trainer in trainers.items():
        trainer.save(str(save_dir / f"{op_name}_final"))
        print(f"  Saved final policy for '{op_name}' to {save_dir}")

    with open(save_dir / "policies.pkl", "wb") as f:
        pickle.dump(dict(policies), f)


def main() -> None:
    """RL training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        type=str,
        default="nut_assembly",
        choices=list(_SYSTEMS),
        help="TAMP system to train on",
    )
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument(
        "--max-rl-steps",
        type=int,
        default=50,
        help="Max RL steps per operator activation",
    )
    parser.add_argument("--operator-predictor", type=str, default=None)
    parser.add_argument(
        "--sigma-threshold",
        type=float,
        default=0.95,
        help="P(success) threshold passed to find_sigma_threshold()",
    )
    parser.add_argument("--save-dir", type=str, default="trained_policies")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
