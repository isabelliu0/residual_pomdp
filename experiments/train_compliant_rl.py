"""Train per-operator compliant RL policies for any TabletopBaseSystem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import imageio
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from pybullet_helpers.camera import capture_image

from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.benchmarks.tabletop_view_occlusion_system import (
    TabletopViewOcclusionTAMPSystem,
)
from residual_controllers.operating_region.data_collect import (
    _check_operator_success,
    _do_nbv_steps,
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


class _Tee:
    """Mirror stdout to both terminal and a log file."""

    def __init__(self, log_path: Path) -> None:
        self._terminal: TextIO = sys.stdout
        self._log = log_path.open("w", buffering=1, encoding="utf-8")

    def write(self, message: str) -> None:
        """Write message to both terminal and log file."""
        self._terminal.write(message)
        self._log.write(message)

    def flush(self) -> None:
        """Flush both terminal and log file."""
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        """Close the log file and restore original stdout."""
        sys.stdout = self._terminal
        self._log.close()


def _find_latest_checkpoint(resume_dir: Path, op_name: str) -> str | None:
    """Return the base path of the latest checkpoint for op_name, or None."""
    final = resume_dir / f"{op_name}_final_policy.zip"
    if final.exists():
        return str(resume_dir / f"{op_name}_final")

    checkpoints = sorted(
        resume_dir.glob(f"{op_name}_ep*_policy.zip"),
        key=lambda p: int(p.stem.split("_ep")[1].split("_")[0]),
    )
    if checkpoints:
        return str(checkpoints[-1])[: -len("_policy.zip")]
    return None


def execute_op_with_phase_checks(
    system: Any, plan: Any, action_str: str, op_goal: set, capture_fn: Any = None
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
        if capture_fn is not None:
            capture_fn()
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
    action_penalty_coef: float = 0.05,
    capture_fn: Any = None,
) -> tuple[bool, bool]:
    """Run RL loop for one operator.

    Returns (succeeded, env_terminated).
    """
    all_labels = system.env.object_labels
    feats0 = extract_features(system.env.belief, all_labels, rel_labels)
    sigma_0 = feats0.relevant_sigma
    sigma_prev = sigma_0
    print(f"    RL start: sigma_0={sigma_0:.10f}, sigma_thresh={sigma_thresh:.10f}")

    for step in range(max_steps):
        obs = encode_belief_for_rl(system.env.belief, rel_labels, n_fold=n_fold)
        action7 = policy.sample_action(obs)
        env_action7 = action7 * policy.action_scale
        system.env.step(np.append(env_action7, 0.0).astype(np.float32))
        if capture_fn is not None:
            capture_fn()
        next_obs = encode_belief_for_rl(system.env.belief, rel_labels, n_fold=n_fold)

        feats = extract_features(system.env.belief, all_labels, rel_labels)
        sigma_curr = feats.relevant_sigma
        success = sigma_curr <= sigma_thresh
        reward = trainer.compute_sigma_reward(
            sigma_prev,
            sigma_curr,
            sigma_thresh,
            sigma_0,
            action=action7,
            action_penalty_coef=action_penalty_coef,
        )
        print(
            f"    RL step {step}: sigma={sigma_curr:.6f}, reward={reward:.4f}, success={success}, margins={sigma_curr - sigma_thresh:.8f})"  # pylint: disable=line-too-long
        )
        trainer.store_transition(obs, action7, reward, next_obs, done=success)
        if trainer.should_train():
            trainer.train_step()

        sigma_prev = sigma_curr

        if success:
            obs = system.env.get_observation()
            current_atoms = (
                system.abstractor.step(obs).atoms if system.abstractor else set()
            )
            print(f"    RL success: current_atoms={current_atoms}")
            print(f"    RL success: op_goal={op_goal}")
            try:
                plan2 = system.plan_for_goal(op_goal, seed=seed)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"    plan_for_goal raised: {e}")
                return False, False
            if plan2 is None:
                print("    plan_for_goal returned None")
                return False, False
            env_terminated = False
            for action in plan2.actions:
                _, _, env_terminated, _, _ = system.env.step(action)
                if capture_fn is not None:
                    capture_fn()
                if env_terminated:
                    break
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

    log_path = save_dir / "train.log"
    tee = _Tee(log_path)
    sys.stdout = tee

    config = vars(args)
    config["log_file"] = str(log_path)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config: {config}")

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
        "action_scale": args.action_scale,
        "zero_init_actor": True,
    }
    trainer_cfg = {"gradient_steps": 1, "train_freq": 1, "min_buffer_size": 256}

    policies: dict[str, RLPolicy] = {}
    trainers: dict[str, RLTrainer] = {}

    episode_successes = 0
    rl_episodes_total = 0
    rl_episode_successes = 0

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

            record_this_ep = (ep + 1) % args.save_freq == 0
            video_frames: list = []

            def _capture_frame() -> None:
                frame = capture_image(
                    system.env.physics_client_id,
                    camera_distance=0.5,
                    camera_yaw=90,
                    camera_pitch=-25,
                    camera_target=(0.5, 0.0, 0.15),
                    image_width=640,
                    image_height=480,
                )
                video_frames.append(frame)  # pylint: disable=cell-var-from-loop

            episode_success = False
            rl_triggered_this_ep = False
            for subseq_actions, obj_names in subsequences:
                rel_labels = list(
                    system.get_object_labels_for_names(obj_names).values()
                )
                obs_dim = len(rel_labels) * 14 + 1

                n_nbv = int(np.random.choice(args.nbv_choices))
                if n_nbv > 0 and rel_labels:
                    print(f"  Running {n_nbv} NBV steps...")
                    _do_nbv_steps(system, n_nbv, rel_labels, episode_seed)

                subseq_goal, _ = system.get_subsequence_effects(subseq_actions)
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
                        if args.resume_dir:
                            ckpt = _find_latest_checkpoint(
                                Path(args.resume_dir), op_name
                            )
                            if ckpt:
                                trainers[op_name].load(ckpt)
                                print(f"  Resumed '{op_name}' from checkpoint {ckpt}")
                            else:
                                print(
                                    f"  No checkpoint found for '{op_name}' in "
                                    f"{args.resume_dir}, starting fresh."
                                )
                        else:
                            print(
                                f"  Created policy for operator '{op_name}' (obs_dim={obs_dim})"  # pylint: disable=line-too-long
                            )

                    sigma_thresh = max(sigma_thresholds.get(op_name, 0.0), 1e-16)

                    if plan_actions is None:
                        plan = system.plan_for_goal(subseq_goal, seed=episode_seed)
                        if plan is None:
                            print(
                                f"  Planning failed for subsequence '{subseq_actions}', skipping."  # pylint: disable=line-too-long
                            )
                            break
                        plan_actions = list(plan.actions)
                        plan_states = list(plan.states)
                        action_idx = 0

                    op_plan = SimpleNamespace(
                        actions=plan_actions[action_idx:],
                        states=plan_states[action_idx:],
                    )
                    phase_failure, env_terminated, consumed = (
                        execute_op_with_phase_checks(
                            system,
                            op_plan,
                            action_str,
                            op_goal,
                            capture_fn=_capture_frame if record_this_ep else None,
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
                            f"(sigma_thresh={sigma_thresh:.10f})"
                        )
                        rl_triggered_this_ep = True
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
                            action_penalty_coef=args.action_penalty_coef,
                            capture_fn=_capture_frame if record_this_ep else None,
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
            if rl_triggered_this_ep:
                rl_episodes_total += 1
                if episode_success:
                    rl_episode_successes += 1

            rl_suffix = (
                f", RL-activated: {rl_episode_successes}/{rl_episodes_total}"
                if rl_triggered_this_ep
                else ""
            )
            print(
                f"  Episode {'SUCCESS' if episode_success else 'FAILURE'} "
                f"({episode_successes}/{ep + 1}{rl_suffix})"
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
                if video_frames:
                    video_path = save_dir / f"ep{ep + 1}.mp4"
                    imageio.mimwrite(str(video_path), video_frames, fps=10)
                    print(f"  Video saved to {video_path}")
                print(f"  Checkpoints saved to {save_dir}")

    finally:
        system.close()

    print("\n=== Training complete ===")
    print(f"Success rate: {episode_successes}/{args.num_episodes}")
    print(f"RL-activated success rate: {rl_episode_successes}/{rl_episodes_total}")
    for op_name, trainer in trainers.items():
        trainer.save(str(save_dir / f"{op_name}_final"))
        print(f"  Saved final policy for '{op_name}' to {save_dir}")

    tee.close()


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
    parser.add_argument("--num-episodes", type=int, default=50)
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
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Directory of a previous training run to resume from",
    )
    parser.add_argument("--save-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.05,
        help="Scale factor applied to RL actions (smaller = gentler exploration)",
    )
    parser.add_argument(
        "--action-penalty-coef",
        type=float,
        default=0.0,
        help="L2 penalty coefficient on unscaled actions to encourage local recovery",
    )
    parser.add_argument(
        "--nbv-choices",
        type=int,
        nargs="+",
        default=[0, 5, 10],
        help="NBV step counts to sample from each subsequence (e.g. --nbv-choices 0 5 10)",  # pylint: disable=line-too-long
    )
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
