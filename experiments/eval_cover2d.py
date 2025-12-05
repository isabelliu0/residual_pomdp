"""Evaluate trained residual RL policies on Cover2D."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from residual_controllers.envs.cover2d import (
    Cover2DConfig,
    Cover2DEnv,
    PickController,
    PlaceController,
    get_mean_state,
)
from residual_controllers.envs.cover2d_residual import (
    action_from_residual,
    encode_belief_cover2d,
)
from residual_controllers.residual_policy import ResidualPolicy


def evaluate_residual_policies(
    pick_model_path: str,
    place_model_path: str,
    num_episodes: int = 100,
    max_steps_per_episode: int = 100,
    video_dir: str = "videos/eval",
    video_freq: int = 10,
    seed: int = 0,
    use_residual: bool = True,
    deterministic: bool = True,
) -> None:
    """Evaluate trained residual policies."""
    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)

    observation_dim = 9
    action_dim = 2

    pick_residual_policy = None
    place_residual_policy = None

    if use_residual:
        pick_residual_policy = ResidualPolicy(
            skill_name="pick",
            observation_dim=observation_dim,
            action_dim=action_dim,
            backend="td3",
            learning_rate=3e-4,
            noise_std=0.1,
            gamma=0.99,
            buffer_size=100000,
            device="cpu",
            seed=seed,
        )
        pick_residual_policy.load(pick_model_path)

        place_residual_policy = ResidualPolicy(
            skill_name="place",
            observation_dim=observation_dim,
            action_dim=action_dim,
            backend="td3",
            learning_rate=3e-4,
            noise_std=0.1,
            gamma=0.99,
            buffer_size=100000,
            device="cpu",
            seed=seed + 1,
        )
        place_residual_policy.load(place_model_path)

    config = Cover2DConfig(
        seed=seed,
        num_particles=10,
        transition_noise_std=0.3,
    )

    print("=" * 60)
    print("Evaluating Residual RL Policies on Cover2D")
    print("=" * 60)
    print(f"Pick model: {pick_model_path}")
    print(f"Place model: {place_model_path}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
    print(f"Use residual: {use_residual}, Deterministic: {deterministic}")
    print(f"Video directory: {video_path}")
    print("-" * 60)

    total_steps = 0
    success_count = 0
    episode_rewards = []
    episode_step_counts = []

    for episode in range(num_episodes):
        env = Cover2DEnv(config)
        belief, _ = env.reset()

        pick_controller = PickController(env.world, target_object_id=0)
        place_controller = PlaceController(
            env.world,
            target_x=env.world.config.goal_region_x
            + env.world.config.goal_region_width / 2,
            target_y=env.world.config.goal_region_y
            + env.world.config.goal_region_height / 2,
        )

        episode_reward = 0.0
        episode_steps = 0
        picked = False

        record_video = episode % video_freq == 0
        if record_video:
            fig, ax = plt.subplots(figsize=(10, 6))
            writer = FFMpegWriter(fps=10)
            mode_str = "residual" if use_residual else "base"
            det_str = "det" if deterministic else "stoch"
            video_file = video_path / f"eval_{mode_str}_{det_str}_ep{episode:04d}.mp4"
            writer.setup(fig, str(video_file), dpi=100)
            env.render(ax=ax, show_belief=True)
            writer.grab_frame()

        for step in range(max_steps_per_episode):
            mean_state = get_mean_state(belief)

            if not mean_state.gripper_state.is_holding and not picked:
                base_action = pick_controller.get_action(belief)

                if use_residual and pick_residual_policy is not None:
                    obs = encode_belief_cover2d(belief)
                    residual = pick_residual_policy.predict(obs, deterministic=deterministic)
                    action = action_from_residual(base_action, residual)
                    actual_residual = np.array(
                        [action.dx - base_action.dx, action.dy - base_action.dy]
                    )
                else:
                    action = base_action
                    actual_residual = None

                belief, _, _, _ = env.step(action)
                mean_state = get_mean_state(belief)
                episode_steps += 1
                total_steps += 1

                if record_video:
                    env.render(ax=ax, show_belief=True, residual_action=actual_residual)
                    writer.grab_frame()

                if mean_state.gripper_state.is_holding:
                    picked = True

            else:
                base_action = place_controller.get_action(belief)

                if use_residual and place_residual_policy is not None:
                    obs = encode_belief_cover2d(belief)
                    residual = place_residual_policy.predict(obs, deterministic=deterministic)
                    action = action_from_residual(base_action, residual)
                    actual_residual = np.array(
                        [action.dx - base_action.dx, action.dy - base_action.dy]
                    )
                else:
                    action = base_action
                    actual_residual = None

                belief, reward, terminal, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if record_video:
                    env.render(ax=ax, show_belief=True, residual_action=actual_residual)
                    writer.grab_frame()

                if terminal:
                    success_count += 1
                    print(f"  [Episode {episode + 1}, Step {step}] SUCCESS!")
                    break

        if record_video:
            writer.finish()
            plt.close(fig)
            print(f"  Video saved to {video_file}")

        episode_rewards.append(episode_reward)
        episode_step_counts.append(episode_steps)

        if (episode + 1) % 10 == 0:
            success_rate = success_count / (episode + 1)
            avg_reward = np.mean(episode_rewards)
            avg_steps = np.mean(episode_step_counts)
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Success Rate: {success_rate:.2%} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Steps: {avg_steps:.1f}"
            )

    success_rate = success_count / num_episodes
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_step_counts)
    std_steps = np.std(episode_step_counts)

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f} ± {std_steps:.1f}")
    print(f"Total Steps: {total_steps}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate residual RL policies on Cover2D")
    parser.add_argument(
        "--pick-model",
        type=str,
        required=True,
        help="Path to pick residual policy model",
    )
    parser.add_argument(
        "--place-model",
        type=str,
        required=True,
        help="Path to place residual policy model",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max steps per episode"
    )
    parser.add_argument(
        "--video-dir", type=str, default="videos/eval", help="Video directory"
    )
    parser.add_argument(
        "--video-freq", type=int, default=10, help="Video recording frequency"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Evaluate base controllers without residual",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (non-deterministic)",
    )

    args = parser.parse_args()

    evaluate_residual_policies(
        pick_model_path=args.pick_model,
        place_model_path=args.place_model,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        video_dir=args.video_dir,
        video_freq=args.video_freq,
        seed=args.seed,
        use_residual=not args.no_residual,
        deterministic=not args.stochastic,
    )
