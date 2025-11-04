"""Residual policy using Stable-Baselines3 backend."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise


class ResidualPolicy:
    """Residual policy for a specific skill using SB3 backend.

    This is a lightweight wrapper around SB3's TD3/DDPG for online
    learning.
    """

    def __init__(
        self,
        skill_name: str,
        observation_dim: int,
        action_dim: int,
        backend: Literal["td3", "ddpg"] = "td3",
        learning_rate: float = 3e-4,
        noise_std: float = 0.1,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        device: str = "cpu",
        seed: int = 0,
    ):
        """Initialize residual policy."""
        self.skill_name = skill_name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.backend_name = backend

        # We create a dummy environment just to initialize SB3
        # The actual training will happen via the replay buffer
        class DummyEnv(gym.Env):  # pylint: disable=abstract-method
            """Minimal env for SB3 initialization."""

            def __init__(self, obs_dim: int, act_dim: int):
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

        dummy_env = DummyEnv(observation_dim, action_dim)

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
        self.model.set_logger(configure(None, ["stdout"]))

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Predict residual action for given observation."""
        observation = np.asarray(observation, dtype=np.float32)
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32).flatten()

    def train(self, gradient_steps: int = 1) -> dict[str, float]:
        """Train the policy for gradient_steps using its replay buffer."""
        self.model.train(gradient_steps=gradient_steps)

        metrics = {}
        if hasattr(self.model, "logger") and self.model.logger is not None:
            for key in ["train/actor_loss", "train/critic_loss", "train/ent_coef"]:
                if key in self.model.logger.name_to_value:
                    metrics[key.replace("train/", "")] = (
                        self.model.logger.name_to_value[key]
                    )

        return metrics

    def add_to_replay_buffer(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ):
        """Add a transition to the replay buffer."""
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action_arr = np.asarray(action, dtype=np.float32).reshape(1, -1)
        reward_arr = np.array([reward], dtype=np.float32)
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
        done_arr = np.array([done], dtype=np.float32)
        self.model.replay_buffer.add(
            obs_arr, next_obs_arr, action_arr, reward_arr, done_arr, [{}]
        )

    def buffer_size(self) -> int:
        """Get current replay buffer size."""
        return self.model.replay_buffer.size()

    def save(self, filepath: str):
        """Save policy to file."""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load policy from file."""
        self.model = self.model.load(filepath, device=self.model.device)
