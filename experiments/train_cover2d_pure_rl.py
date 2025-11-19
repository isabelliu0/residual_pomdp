"""Train pure RL baseline for Cover2D with full action space."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from residual_controllers.envs.cover2d import (
    Action,
    Cover2DConfig,
    Cover2DEnv,
    GripperAction,
)
from residual_controllers.envs.cover2d_residual import encode_belief_cover2d
from residual_controllers.online_trainer import OnlineTrainer
from residual_controllers.residual_policy import ResidualPolicy


def train_pure_rl(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 200,
    save_dir: str = "trained_models",
    video_dir: str = "videos",
    video_freq: int = 10,
    seed: int = 0,
    record_video: bool = False,
) -> None:
    """Train pure RL policy for Cover2DEnv with full action space."""
    run_name = f"pure_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_dir) / run_name
    if record_video:
        video_path.mkdir(parents=True, exist_ok=True)

    config = Cover2DConfig(
        seed=seed,
        num_particles=10,
        transition_noise_std=0.3,
    )

    observation_dim = 9
    action_dim = 4

    policy = ResidualPolicy(
        skill_name="pure_rl",
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

    trainer = OnlineTrainer(
        residual_policy=policy,
        gradient_steps=1,
        train_freq=1,
        min_buffer_size=256,
    )

    print("Training pure RL baseline for Cover2D")
    print(f"Run name: {run_name}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
    print(f"Observation dim: {observation_dim}, Action dim: {action_dim}")
    print(f"Save directory: {save_path}")
    print(f"Video directory: {video_path}")
    print("-" * 60)

    total_steps = 0
    success_count = 0

    for episode in range(num_episodes):
        env = Cover2DEnv(config)
        belief, _ = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        record_this_episode = record_video and (episode % video_freq == 0)
        if record_this_episode:
            fig, ax = plt.subplots(figsize=(10, 6))
            writer = FFMpegWriter(fps=10)
            video_file = video_path / f"episode_{episode:04d}.mp4"
            writer.setup(fig, str(video_file), dpi=100)
            env.render(ax=ax, show_belief=True)
            writer.grab_frame()

        for _ in range(max_steps_per_episode):
            obs = encode_belief_cover2d(belief)
            action_array = policy.predict(obs, deterministic=False)

            dx = float(np.clip(action_array[0], -0.3, 0.3))
            dy = float(np.clip(action_array[1], -0.3, 0.3))
            dtheta = float(np.clip(action_array[2], -0.2, 0.2))
            gripper_logit = float(action_array[3])

            if gripper_logit < -0.33:  # action space: [-1, 1]
                gripper_action = GripperAction.NOOP
            elif gripper_logit < 0.33:
                gripper_action = GripperAction.PICK
            else:
                gripper_action = GripperAction.PLACE

            action = Action(dx=dx, dy=dy, dtheta=dtheta, gripper_action=gripper_action)

            belief, reward, terminal, _ = env.step(action)
            next_obs = encode_belief_cover2d(belief)

            trainer.store_transition(
                obs,
                action_array,
                reward,
                next_obs,
                terminal,
            )

            trainer.train_step()

            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if record_this_episode:
                env.render(ax=ax, show_belief=True)
                writer.grab_frame()

            if terminal:
                success_count += 1
                break

        if record_this_episode:
            writer.finish()
            plt.close(fig)
            print(f"  Video saved: {video_file}")

        if episode % 10 == 0:
            success_rate = success_count / (episode + 1)
            print(
                f"Episode {episode:4d} | Steps: {episode_steps:3d} | "
                f"Reward: {episode_reward:6.2f} | Success rate: {success_rate:.2%} | "
                f"Total steps: {total_steps}"
            )

        if episode % 100 == 0 and episode > 0:
            model_path = save_path / f"model_episode_{episode}.pkl"
            policy.save(str(model_path))
            print(f"Model saved: {model_path}")

    final_model_path = save_path / "final_model.pkl"
    policy.save(str(final_model_path))
    print(f"\nTraining complete! Final model saved: {final_model_path}")
    print(f"Final success rate: {success_count / num_episodes:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pure RL baseline for Cover2D")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--video-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")

    args = parser.parse_args()

    train_pure_rl(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        video_dir=args.video_dir,
        video_freq=args.video_freq,
        seed=args.seed,
        record_video=args.record_video,
    )
