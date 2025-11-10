"""Tests for ResidualPolicy class."""

import numpy as np

from residual_controllers.residual_policy import ResidualPolicy


def test_residual_policy_creation():
    """Test creation of ResidualPolicy instance."""
    policy = ResidualPolicy(
        skill_name="test_skill",
        observation_dim=10,
        action_dim=2,
        backend="td3",
        device="cpu",
    )
    assert policy.skill_name == "test_skill"
    assert policy.observation_dim == 10
    assert policy.action_dim == 2


def test_residual_policy_predict():
    """Test action prediction."""
    policy = ResidualPolicy(
        skill_name="test_skill",
        observation_dim=10,
        action_dim=2,
        backend="td3",
        device="cpu",
    )
    obs = np.random.randn(10).astype(np.float32)
    action = policy.predict(obs, deterministic=True)
    assert action.shape == (2,)
    assert action.dtype == np.float32


def test_residual_policy_buffer():
    """Test adding to replay buffer."""
    policy = ResidualPolicy(
        skill_name="test_skill",
        observation_dim=10,
        action_dim=2,
        backend="td3",
        device="cpu",
    )
    obs = np.random.randn(10).astype(np.float32)
    action = np.random.randn(2).astype(np.float32)
    next_obs = np.random.randn(10).astype(np.float32)

    policy.add_to_replay_buffer(obs, action, 1.0, next_obs, False)
    assert policy.buffer_size() == 1
