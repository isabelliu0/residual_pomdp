"""Train Cover2D residual RL with Domain Randomization."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from residual_controllers.envs.cover2d import (
    Action,
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


class DomainRandomizedEnv:
    """Wrapper that applies domain randomization to Cover2DEnv."""

    def __init__(
        self,
        config: Cover2DConfig,
        transition_noise_range: tuple[float, float] = (0.2, 0.5),
        action_scale_range: tuple[float, float] = (0.8, 1.2),
        rotation_noise_scale_range: tuple[float, float] = (0.3, 0.7),
        rng: np.random.Generator | None = None,
    ):
        self.base_config = config
        self.transition_noise_range = transition_noise_range
        self.action_scale_range = action_scale_range
        self.rotation_noise_scale_range = rotation_noise_scale_range
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

        self.env = None
        self.action_scale = 1.0
        self.rotation_noise_scale = 0.5

    def reset(self):
        """Reset with new randomized parameters."""
        transition_noise = self.rng.uniform(*self.transition_noise_range)
        self.action_scale = self.rng.uniform(*self.action_scale_range)
        self.rotation_noise_scale = self.rng.uniform(*self.rotation_noise_scale_range)

        config = replace(
            self.base_config,
            transition_noise_std=transition_noise,
            seed=self.rng.integers(0, 1000000),
        )

        self.env = Cover2DEnv(config)
        return self.env.reset()

    def step(self, action: Action):
        """Apply action with scaling."""
        scaled_action = Action(
            dx=action.dx * self.action_scale,
            dy=action.dy * self.action_scale,
            dtheta=action.dtheta * self.rotation_noise_scale,
            gripper_action=action.gripper_action,
        )
        return self.env.step(scaled_action)

    def render(self, **kwargs):
        """Render the environment."""
        return self.env.render(**kwargs)

    @property
    def world(self):
        """Get world from underlying environment."""
        return self.env.world

    @property
    def steps(self):
        """Get steps from underlying environment."""
        return self.env.steps


def train_with_domain_randomization(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 100,
    save_dir: str = "trained_models",
    video_dir: str = "videos",
    video_freq: int = 50,
    seed: int = 0,
) -> None:
    """Train residual RL with domain randomization."""
    run_name = f"dr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_dir) / run_name
    video_path.mkdir(parents=True, exist_ok=True)

    base_config = Cover2DConfig(
        seed=seed,
        num_particles=10,
        transition_noise_std=0.3,
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

    print("Training residual RL with Domain Randomization on Cover2D")
    print(f"Run name: {run_name}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps_per_episode}")
    print(f"Observation dim: {observation_dim}, Action dim: {action_dim}")
    print(f"Domain Randomization:")
    print(f"  - Transition noise std: [0.2, 0.5]")
    print(f"  - Action scale: [0.8, 1.2]")
    print(f"  - Rotation noise scale: [0.3, 0.7]")
    print(f"Save directory: {save_path}")
    print(f"Video directory: {video_path}")
    print("-" * 60)

    total_steps = 0
    success_count = 0

    dr_env = DomainRandomizedEnv(base_config, rng=np.random.default_rng(seed))

    for episode in range(num_episodes):
        belief, _ = dr_env.reset()

        pick_controller = PickController(dr_env.world, target_object_id=0)
        place_controller = PlaceController(
            dr_env.world,
            target_x=dr_env.world.config.goal_region_x
            + dr_env.world.config.goal_region_width / 2,
            target_y=dr_env.world.config.goal_region_y
            + dr_env.world.config.goal_region_height / 2,
        )

        episode_reward = 0.0
        episode_steps = 0
        picked = False
        placed = False

        record_video = episode % video_freq == 0
        if record_video:
            fig, ax = plt.subplots(figsize=(10, 6))
            writer = FFMpegWriter(fps=10)
            video_file = video_path / f"episode_{episode:04d}.mp4"
            writer.setup(fig, str(video_file), dpi=100)
            dr_env.render(ax=ax, show_belief=True)
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

                belief, _, _, _ = dr_env.step(action)
                next_obs = encode_belief_cover2d(belief)

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
                    dr_env.render(ax=ax, show_belief=True, residual_action=actual_residual)
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

                belief, reward, terminal, _ = dr_env.step(action)
                next_obs = encode_belief_cover2d(belief)

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
                    dr_env.render(ax=ax, show_belief=True, residual_action=actual_residual)
                    writer.grab_frame()

                if terminal:
                    placed = True
                    success_count += 1
                    print(
                        f"  [Episode {episode + 1}, Step {step}] SUCCESS! "
                        f"Object placed in goal!"
                    )
                    break

        if record_video:
            writer.finish()
            plt.close(fig)
            print(f"  Video saved to {video_file}")

        pick_stats = pick_trainer.get_training_stats()
        place_stats = place_trainer.get_training_stats()
        success_rate = success_count / (episode + 1)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Steps: {episode_steps} | "
            f"Reward: {episode_reward:.2f} | "
            f"Success: {placed} | "
            f"Success Rate: {success_rate:.2%} | "
            f"Pick Buffer: {pick_stats['num_transitions']} | "
            f"Place Buffer: {place_stats['num_transitions']} | "
            f"DR Params: noise={dr_env.env.world.config.transition_noise_std:.2f}, "
            f"action_scale={dr_env.action_scale:.2f}, "
            f"rot_noise_scale={dr_env.rotation_noise_scale:.2f}"
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
        description="Train residual RL with Domain Randomization on Cover2D"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max steps per episode"
    )
    parser.add_argument(
        "--save-dir", type=str, default="trained_models", help="Save directory"
    )
    parser.add_argument(
        "--video-dir", type=str, default="videos", help="Video directory"
    )
    parser.add_argument(
        "--video-freq", type=int, default=50, help="Video recording frequency"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    train_with_domain_randomization(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        video_dir=args.video_dir,
        video_freq=args.video_freq,
        seed=args.seed,
    )
