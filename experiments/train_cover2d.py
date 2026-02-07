"""Train Cover2D residual RL policies for both Pick and Place controllers."""

from __future__ import annotations

import argparse
from datetime import datetime
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
from residual_controllers.online_trainer import OnlineTrainer
from residual_controllers.residual_policy import ResidualPolicy


def train_residual(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 100,
    save_dir: str = "trained_models",
    video_dir: str = "videos",
    video_freq: int = 500,
    seed: int = 0,
    num_beacons: int = 3,
    info_bonus_weight: float = 0.3,
    info_alpha: float = 1.0,
) -> None:
    """Train residual RL policy for PlaceController on Cover2DEnv."""
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_dir) / run_name
    video_path.mkdir(parents=True, exist_ok=True)

    config = Cover2DConfig(
        seed=seed,
        num_particles=10,
        transition_noise_std=0.3,
        num_beacons=num_beacons,
        info_bonus_weight=info_bonus_weight,
        info_alpha=info_alpha,
    )

    observation_dim = 9
    action_dim = 2

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

    pick_trainer = OnlineTrainer(
        residual_policy=pick_residual_policy,
        gradient_steps=1,
        train_freq=1,
        min_buffer_size=256,
    )

    place_trainer = OnlineTrainer(
        residual_policy=place_residual_policy,
        gradient_steps=1,
        train_freq=1,
        min_buffer_size=256,
    )

    print("Training residual RL for Pick and Place Controllers on Cover2D")
    print(f"Run name: {run_name}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
    print(f"Observation dim: {observation_dim}, Action dim: {action_dim}")
    print(
        f"Beacons: {num_beacons}, Info bonus weight: {info_bonus_weight}, Info alpha: {info_alpha}"  # pylint: disable=line-too-long
    )
    print(f"Save directory: {save_path}")
    print(f"Video directory: {video_path}")
    print("-" * 60)

    total_steps = 0
    success_count = 0

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
        episode_task_reward = 0.0
        episode_info_reward = 0.0
        episode_steps = 0
        picked = False
        placed = False
        trans_uncertainties = []
        ang_uncertainties = []

        record_video = episode % video_freq == 0
        if record_video:
            fig, ax = plt.subplots(figsize=(10, 6))
            writer = FFMpegWriter(fps=10)
            video_file = video_path / f"episode_{episode:04d}.mp4"
            writer.setup(fig, str(video_file), dpi=100)
            env.render(ax=ax, show_belief=True)
            writer.grab_frame()

        for step in range(max_steps_per_episode):
            mean_state = get_mean_state(belief)

            if not mean_state.gripper_state.is_holding and not picked:
                base_action = pick_controller.get_action(belief)
                obs = encode_belief_cover2d(belief)
                residual = pick_residual_policy.predict(obs, deterministic=False)
                action = action_from_residual(base_action, residual)

                actual_residual = np.array(
                    [action.dx - base_action.dx, action.dy - base_action.dy]
                )

                belief, _, _, info = env.step(action)
                next_obs = encode_belief_cover2d(belief)

                trans_uncertainties.append(info["translational_uncertainty"])
                ang_uncertainties.append(info["angular_uncertainty"])
                episode_task_reward += info["task_reward"]
                episode_info_reward += info["info_reward"]

                mean_state = get_mean_state(belief)
                pick_reward = 1.0 if mean_state.gripper_state.is_holding else 0.0
                pick_terminal = mean_state.gripper_state.is_holding

                pick_trainer.store_transition(
                    obs,
                    residual,
                    pick_reward,
                    next_obs,
                    pick_terminal,
                )

                if pick_trainer.should_train():
                    _ = pick_trainer.train_step()

                episode_reward += pick_reward
                episode_steps += 1
                total_steps += 1

                if record_video:
                    env.render(ax=ax, show_belief=True, residual_action=actual_residual)
                    writer.grab_frame()

                if mean_state.gripper_state.is_holding:
                    picked = True

            else:
                base_action = place_controller.get_action(belief)
                obs = encode_belief_cover2d(belief)
                residual = place_residual_policy.predict(obs, deterministic=False)
                action = action_from_residual(base_action, residual)

                actual_residual = np.array(
                    [action.dx - base_action.dx, action.dy - base_action.dy]
                )

                belief, reward, terminal, info = env.step(action)
                next_obs = encode_belief_cover2d(belief)

                trans_uncertainties.append(info["translational_uncertainty"])
                ang_uncertainties.append(info["angular_uncertainty"])
                episode_task_reward += info["task_reward"]
                episode_info_reward += info["info_reward"]

                place_trainer.store_transition(
                    obs,
                    residual,
                    reward,
                    next_obs,
                    terminal,
                )

                if place_trainer.should_train():
                    _ = place_trainer.train_step()

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if record_video:
                    env.render(ax=ax, show_belief=True, residual_action=actual_residual)
                    writer.grab_frame()

                if terminal:
                    placed = True
                    success_count += 1
                    print(
                        f"  [Episode {episode + 1}, Step {step}] SUCCESS! Object placed in goal!"  # pylint: disable=line-too-long
                    )
                    break

        if record_video:
            writer.finish()
            plt.close(fig)
            print(f"  Video saved to {video_file}")

        pick_stats = pick_trainer.get_training_stats()
        place_stats = place_trainer.get_training_stats()
        success_rate = success_count / (episode + 1)

        avg_trans_unc = np.mean(trans_uncertainties) if trans_uncertainties else 0.0
        avg_ang_unc = np.mean(ang_uncertainties) if ang_uncertainties else 0.0

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Steps: {episode_steps} | "
            f"Reward: {episode_reward:.2f} (task: {episode_task_reward:.2f}, info: {episode_info_reward:.2f}) | "  # pylint: disable=line-too-long
            f"Success: {placed} | "
            f"Success Rate: {success_rate:.2%} | "
            f"Avg Uncertainty (trans/ang): {avg_trans_unc:.3f}/{avg_ang_unc:.3f} | "
            f"Pick Buffer: {pick_stats['num_transitions']} | "
            f"Place Buffer: {place_stats['num_transitions']}"
        )

        if (episode + 1) % 10 == 0:
            pick_model_path = save_path / f"residual_pick_ep{episode + 1}.pkl"
            place_model_path = save_path / f"residual_place_ep{episode + 1}.pkl"
            pick_residual_policy.save(str(pick_model_path))
            place_residual_policy.save(str(place_model_path))
            print(f"  Saved checkpoints to {save_path}")

    final_pick_model_path = save_path / "residual_pick_final.pkl"
    final_place_model_path = save_path / "residual_place_final.pkl"
    pick_residual_policy.save(str(final_pick_model_path))
    place_residual_policy.save(str(final_place_model_path))

    print("-" * 60)
    print("Training completed!")
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Final pick model saved to {final_pick_model_path}")
    print(f"Final place model saved to {final_place_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train residual RL for PlaceController on Cover2D"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max steps per episode"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="trained_models",
        help="Directory to save models",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num-beacons",
        type=int,
        default=3,
        help="Number of beacons for information gathering",
    )
    parser.add_argument(
        "--info-bonus-weight",
        type=float,
        default=1.0,
        help="Weight for information gain reward (lambda)",
    )
    parser.add_argument(
        "--info-alpha",
        type=float,
        default=1.0,
        help="Weight for angular vs translational uncertainty (alpha)",
    )

    args = parser.parse_args()

    train_residual(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        seed=args.seed,
        num_beacons=args.num_beacons,
        info_bonus_weight=args.info_bonus_weight,
        info_alpha=args.info_alpha,
    )
