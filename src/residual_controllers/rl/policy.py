"""Standalone RL policy using Stable-Baselines3 TD3."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise


class _DummyEnv(gym.Env):  # pylint: disable=abstract-method
    """Module-level dummy env for SB3 policy initialization (must be
    picklable)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):  # pylint: disable=unused-argument
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


class RLPolicy:
    """Standalone TD3/DDPG policy for online RL training."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        backend: Literal["td3", "ddpg"] = "td3",
        learning_rate: float = 3e-4,
        noise_std: float = 0.1,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        device: str = "cpu",
        seed: int = 0,
        action_scale: float = 1.0,
        zero_init_actor: bool = False,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_scale = action_scale

        dummy_env = _DummyEnv(observation_dim, action_dim)
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=np.ones(action_dim, dtype=np.float32) * noise_std,
        )
        algo_cls = TD3 if backend == "td3" else DDPG
        self.model = algo_cls(
            "MlpPolicy",
            dummy_env,
            learning_rate=learning_rate,
            action_noise=action_noise,
            gamma=gamma,
            buffer_size=buffer_size,
            device=device,
            seed=seed,
            verbose=0,
        )

        if zero_init_actor:
            for net in (self.model.actor.mu, self.model.actor_target.mu):
                for module in reversed(list(net.modules())):
                    if isinstance(module, torch.nn.Linear):
                        with torch.no_grad():
                            module.weight.zero_()
                            module.bias.zero_()
                        break

        self.model._setup_learn(total_timesteps=100_000_000)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action deterministically (inference only).

        Returns unscaled action in [-1, 1]. Multiply by action_scale
        before sending to the environment.
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).flatten()

    def sample_action(self, observation: np.ndarray) -> np.ndarray:
        """Sample action with exploration noise for training.

        Returns unscaled action in [-1, 1] suitable for storing in the
        replay buffer. Multiply by action_scale before sending to the
        environment.
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).flatten()
        if self.model.action_noise is not None:
            noise = self.model.action_noise()
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def add_to_replay_buffer(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> None:
        """Add transition to replay buffer."""
        assert self.model.replay_buffer is not None
        self.model.replay_buffer.add(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            np.asarray(next_obs, dtype=np.float32).reshape(1, -1),
            np.asarray(action, dtype=np.float32).reshape(1, -1),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [{}],
        )

    def train(self, gradient_steps: int = 1) -> dict[str, float]:
        """Perform training step."""
        self.model.train(gradient_steps=gradient_steps)
        metrics: dict[str, float] = {}
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key in ["train/actor_loss", "train/critic_loss"]:
                if key in self.model.logger.name_to_value:
                    metrics[key.replace("train/", "")] = float(
                        self.model.logger.name_to_value[key]
                    )
        return metrics

    def buffer_size(self) -> int:
        """Return current size of replay buffer."""
        assert self.model.replay_buffer is not None
        return self.model.replay_buffer.size()

    def save(self, filepath: str) -> None:
        """Save model to filepath."""
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from filepath."""
        self.model = self.model.load(filepath, device=self.model.device)
