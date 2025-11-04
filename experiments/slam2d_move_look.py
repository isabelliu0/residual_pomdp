"""Main script for training residual policy on SLAM2D move_look skill.

This implements online residual RL by:
1. Running many different SLAM2D tasks
2. For each task, executing plans with SymK replanning
3. Training the residual policy on move_look executions
"""

from __future__ import annotations

import random
from pathlib import Path

from tampura.structs import Action
from tampura.symbolic import Atom
from tampura_environments.slam_collect.env import (
    SlamCollectEnv,
    move_pick,
    move_place,
    rrt_with_action_noise,
)

from residual_controllers.executor import execute_action_sequence
from residual_controllers.online_trainer import OnlineTrainer
from residual_controllers.planner import (
    filter_plans_with_skill,
    get_base_action_name,
    get_plans,
    parse_verified_effects,
)
from residual_controllers.residual_policy import ResidualPolicy
from residual_controllers.utils import encode_slam_belief, select_mean_particle


def inject_move_look_actions(plan: list[Action], store) -> list[Action]:
    """Inject move_look actions before every move_pick."""
    modified_plan = []
    all_regions = store.type_dict.get("region", [])
    beacon_aliases = [r for r in all_regions if r.startswith("o_beacon")]
    for action in plan:
        action_base_name = get_base_action_name(action.name)
        if action_base_name == "move_pick":
            if beacon_aliases:
                move_look_action = Action(
                    name="move_look_cm_0.1.0",
                    args=[
                        beacon_aliases[0]
                    ],  # NOTE: Use first beacon deterministically
                )
                modified_plan.append(move_look_action)
        modified_plan.append(action)
    return modified_plan


def move_look_deterministic(action: Action, belief, store):
    """Deterministic version of move_look using select_mean_particle."""
    (region_index,) = store.get_all(action.args)
    particles = belief.particles
    mean_particle = select_mean_particle(belief)
    solved, actions = rrt_with_action_noise(
        [mean_particle], goal=particles[0].regions[region_index], std=0
    )
    if not solved:
        return None
    return actions


def unified_controller(action: Action, belief, store):
    """Unified controller that handles all action types.

    For move_look: use deterministic controller
    For other actions: use original stochastic controllers
    """
    action_base_name = get_base_action_name(action.name)

    if action_base_name == "move_look":
        return move_look_deterministic(action, belief, store)
    if action_base_name == "move_pick":
        return move_pick(action, belief, store)
    if action_base_name == "move_place":
        return move_place(action, belief, store)
    raise ValueError(f"Unknown action type: {action_base_name}")


def run_single_task(env, policy, trainer, TARGET_SKILL, MAX_REPLANS=10):
    """Run a single SLAM2D task with replanning and training."""
    belief, store = env.initialize()
    spec = env.get_problem_spec()

    task_stats = {
        "num_replans": 0,
        "total_actions": 0,
        "move_look_executions": 0,
        "move_look_successes": 0,
        "goal_reached": False,
    }

    for _ in range(MAX_REPLANS):
        abstract_belief = belief.abstract(store)
        all_plans = get_plans(
            spec=spec,
            abstract_belief=abstract_belief,
            store=store,
            num_plans=10,
        )

        if len(all_plans) == 0:
            print("  No plans found, ending task.")
            break

        plans_with_skill = filter_plans_with_skill(all_plans, TARGET_SKILL)
        plan = (
            plans_with_skill[0]
            if len(plans_with_skill) > 0
            else random.choice(all_plans)
        )

        # NOTE: TAMPURA does not include information-gathering actions in top-K plans
        # For now, we manually inject move_look actions before every move_pick
        plan = inject_move_look_actions(plan, store)

        # Execute plan action by action
        plan_success = True
        for symbolic_action in plan:
            task_stats["total_actions"] += 1
            action_base_name = get_base_action_name(symbolic_action.name)
            expected_veffects = parse_verified_effects(symbolic_action.name)

            state_before = None
            if action_base_name == TARGET_SKILL:
                state_before = encode_slam_belief(belief)

            next_belief, actual_veffects, residual_action = execute_action_sequence(
                symbolic_action=symbolic_action,
                belief=belief,
                store=store,
                spec=spec,
                base_controller=unified_controller,
                residual_policy=policy if action_base_name == TARGET_SKILL else None,
                target_skill=TARGET_SKILL,
                belief_encoder=encode_slam_belief,
            )

            # Check if verified effects match
            # NOTE: For non-target skills, ignore known_pose effect
            if action_base_name != TARGET_SKILL and expected_veffects is not None:
                action_schema = spec.get_action_schema(action_base_name)
                known_pose_indices = []
                if hasattr(action_schema, "verify_effects"):
                    for i, veffect in enumerate(action_schema.verify_effects):
                        if (
                            isinstance(veffect, Atom)
                            and veffect.pred_name == "known_pose"
                        ):
                            known_pose_indices.append(i)
                if known_pose_indices:
                    expected_filtered = [
                        e
                        for i, e in enumerate(expected_veffects)
                        if i not in known_pose_indices
                    ]
                    actual_filtered = [
                        a
                        for i, a in enumerate(actual_veffects)
                        if i not in known_pose_indices
                    ]
                    effects_match = len(expected_filtered) == len(
                        actual_filtered
                    ) and all(
                        e == a for e, a in zip(expected_filtered, actual_filtered)
                    )
                    print(
                        f"Checking effects for {action_base_name}: expected {expected_filtered}, actual {actual_filtered}"  # pylint: disable=line-too-long
                    )
                else:
                    effects_match = len(expected_veffects) == len(
                        actual_veffects
                    ) and all(
                        e == a for e, a in zip(expected_veffects, actual_veffects)
                    )
                    print(
                        f"Checking effects for {action_base_name}: expected {expected_veffects}, actual {actual_veffects}"  # pylint: disable=line-too-long
                    )
            else:
                # For TARGET_SKILL, check all verified effects
                effects_match = (
                    expected_veffects is not None
                    and len(expected_veffects) == len(actual_veffects)
                    and all(e == a for e, a in zip(expected_veffects, actual_veffects))
                )
                print(
                    f"Checking effects for {action_base_name}: expected {expected_veffects}, actual {actual_veffects}"  # pylint: disable=line-too-long
                )

            if action_base_name == TARGET_SKILL and residual_action is not None:
                task_stats["move_look_executions"] += 1
                reward = 1.0 if effects_match else 0.0

                if effects_match:
                    task_stats["move_look_successes"] += 1

                state_after = encode_slam_belief(next_belief)

                trainer.store_transition(
                    state=state_before,
                    action=residual_action,
                    reward=reward,
                    next_state=state_after,
                    done=False,
                )

                if trainer.should_train():
                    trainer.train_step()

            if not effects_match:
                plan_success = False
                break

            belief = next_belief

        if plan_success:
            task_stats["goal_reached"] = True
            break

        task_stats["num_replans"] += 1

    return task_stats


def main():
    """Main training loop over multiple SLAM2D tasks."""

    TARGET_SKILL = "move_look"
    NUM_TASKS = 100
    MAX_REPLANS_PER_TASK = 10
    ACTION_DIM = 2  # [delta_dx, delta_dy]

    print("=" * 60)
    print("Training Residual Policy for move_look")
    print("=" * 60)

    config = {
        "vis": False,
        "save_dir": "/tmp/residual_rl_training",
    }

    env = SlamCollectEnv(config=config)
    belief, _ = env.initialize()
    obs_dim = encode_slam_belief(belief).shape[0]
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {ACTION_DIM}")
    print(f"Number of tasks: {NUM_TASKS}")

    policy = ResidualPolicy(
        skill_name=TARGET_SKILL,
        observation_dim=obs_dim,
        action_dim=ACTION_DIM,
        backend="td3",
        learning_rate=3e-4,
        buffer_size=10000,
        device="cpu",
    )

    trainer = OnlineTrainer(
        residual_policy=policy,
        gradient_steps=1,
        train_freq=1,
        min_buffer_size=128,
    )

    total_move_look_executions = 0
    total_move_look_successes = 0
    goals_reached = 0

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for task_idx in range(NUM_TASKS):
        print(f"\n--- Task {task_idx + 1}/{NUM_TASKS} ---")

        task_stats = run_single_task(
            env=env,
            policy=policy,
            trainer=trainer,
            TARGET_SKILL=TARGET_SKILL,
            MAX_REPLANS=MAX_REPLANS_PER_TASK,
        )

        total_move_look_executions += task_stats["move_look_executions"]
        total_move_look_successes += task_stats["move_look_successes"]
        if task_stats["goal_reached"]:
            goals_reached += 1

        print(
            f"  Task completed: {task_stats['move_look_executions']} move_look executions, "  # pylint: disable=line-too-long
            f"{task_stats['move_look_successes']} successes"
        )

        if (task_idx + 1) % 10 == 0:
            stats = trainer.get_training_stats()
            success_rate = (
                100 * total_move_look_successes / total_move_look_executions
                if total_move_look_executions > 0
                else 0
            )
            print(f"\n  [Progress after {task_idx + 1} tasks]")
            print(f"    Total move_look executions: {total_move_look_executions}")
            print(f"    Total successes: {total_move_look_successes}")
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Buffer size: {stats['buffer_size']}")
            print(f"    Training updates: {stats['num_updates']}")
            print(f"    Goals reached: {goals_reached}/{task_idx + 1}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total tasks: {NUM_TASKS}")
    print(
        f"Goals reached: {goals_reached}/{NUM_TASKS} ({100 * goals_reached / NUM_TASKS:.1f}%)"  # pylint: disable=line-too-long
    )
    print(f"Total move_look executions: {total_move_look_executions}")
    print(f"Total successes: {total_move_look_successes}")
    if total_move_look_executions > 0:
        success_rate = 100 * total_move_look_successes / total_move_look_executions
        print(f"Overall success rate: {success_rate:.1f}%")

    final_stats = trainer.get_training_stats()
    print("\nFinal training stats:")
    print(f"  Buffer size: {final_stats['buffer_size']}")
    print(f"  Total updates: {final_stats['num_updates']}")
    if "avg_critic_loss" in final_stats:
        print(f"  Avg critic loss: {final_stats['avg_critic_loss']:.4f}")
    if "avg_q_value" in final_stats:
        print(f"  Avg Q-value: {final_stats['avg_q_value']:.4f}")

    save_path = "trained_models/move_look_residual"
    Path("trained_models").mkdir(exist_ok=True)
    trainer.save(save_path)
    print(f"\nSaved trained policy to {save_path}")


if __name__ == "__main__":
    main()
