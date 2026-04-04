"""Online training wrapper for RLPolicy."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from residual_controllers.rl.policy import RLPolicy


class RLTrainer:
    """Online training of an RLPolicy during execution."""

    def __init__(
        self,
        policy: RLPolicy,
        gradient_steps: int = 1,
        train_freq: int = 1,
        min_buffer_size: int = 256,
    ) -> None:
        self.policy = policy
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.min_buffer_size = min_buffer_size

        self.num_transitions = 0
        self.num_updates = 0
        self.metrics_history: list[dict[str, float]] = []

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store transition in replay buffer."""
        self.policy.add_to_replay_buffer(obs, action, reward, next_obs, done)
        self.num_transitions += 1

    def should_train(self) -> bool:
        """Return True if conditions are met to perform a training step."""
        return (
            self.policy.buffer_size() >= self.min_buffer_size
            and self.num_transitions % self.train_freq == 0
        )

    def train_step(self) -> dict[str, float]:
        """Perform a training step."""
        if not self.should_train():
            return {}
        metrics = self.policy.train(gradient_steps=self.gradient_steps)
        if metrics:
            self.num_updates += 1
            self.metrics_history.append(metrics)
        return metrics

    def get_stats(self) -> dict[str, Any]:
        """Return training statistics."""
        stats: dict[str, Any] = {
            "num_transitions": self.num_transitions,
            "num_updates": self.num_updates,
            "buffer_size": self.policy.buffer_size(),
        }
        if self.metrics_history:
            recent = self.metrics_history[-min(100, len(self.metrics_history)) :]
            for key in ["actor_loss", "critic_loss"]:
                vals = [m[key] for m in recent if key in m]
                if vals:
                    stats[f"avg_{key}"] = float(np.mean(vals))
        return stats

    def save(self, filepath: str) -> None:
        """Save policy and training metadata to filepath."""
        self.policy.save(f"{filepath}_policy.zip")
        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "num_transitions": self.num_transitions,
                    "num_updates": self.num_updates,
                    "metrics_history": self.metrics_history,
                },
                f,
            )

    def load(self, filepath: str) -> None:
        """Load policy and training metadata from filepath."""
        self.policy.load(f"{filepath}_policy.zip")
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            data = pickle.load(f)
        self.num_transitions = data["num_transitions"]
        self.num_updates = data["num_updates"]
        self.metrics_history = data["metrics_history"]
