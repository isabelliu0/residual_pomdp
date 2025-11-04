"""Online training recipe for residual policies."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from residual_controllers.residual_policy import ResidualPolicy


class OnlineTrainer:
    """Online training of residual policy during execution."""

    def __init__(
        self,
        residual_policy: ResidualPolicy,
        gradient_steps: int = 1,
        train_freq: int = 1,
        min_buffer_size: int = 256,
    ):
        """Initialize online trainer."""
        self.policy = residual_policy
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.min_buffer_size = min_buffer_size

        self.num_transitions = 0
        self.num_updates = 0
        self.metrics_history: list[dict[str, float]] = []

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Store a transition in the policy's replay buffer."""
        self.policy.add_to_replay_buffer(state, action, reward, next_state, done)
        self.num_transitions += 1

    def should_train(self) -> bool:
        """Check if we should run a training update."""
        return (
            self.policy.buffer_size() >= self.min_buffer_size
            and self.num_transitions % self.train_freq == 0
        )

    def train_step(self) -> dict[str, float]:
        """Perform training updates on the policy."""
        if not self.should_train():
            return {}
        metrics = self.policy.train(gradient_steps=self.gradient_steps)
        if metrics:
            self.num_updates += 1
            self.metrics_history.append(metrics)
        return metrics

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        stats: dict[str, Any] = {
            "num_transitions": self.num_transitions,
            "num_updates": self.num_updates,
            "buffer_size": self.policy.buffer_size(),
        }
        if len(self.metrics_history) > 0:
            recent_window = min(100, len(self.metrics_history))
            recent_metrics = self.metrics_history[-recent_window:]
            for key in ["actor_loss", "critic_loss", "q_value"]:
                values = [m[key] for m in recent_metrics if key in m]
                if values:
                    stats[f"avg_{key}"] = np.mean(values)
        return stats

    def save(self, filepath: str):
        """Save trainer state (policy + metadata)."""
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
        print(f"Saved trainer to {filepath}")

    def load(self, filepath: str):
        """Load trainer state (policy + metadata)."""
        self.policy.load(f"{filepath}_policy.zip")
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.num_transitions = data["num_transitions"]
            self.num_updates = data["num_updates"]
            self.metrics_history = data["metrics_history"]
        print(f"Loaded trainer from {filepath}")
