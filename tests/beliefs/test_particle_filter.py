"""Tests for particle filter operations."""

import numpy as np

from residual_controllers.beliefs import (
    TabletopState,
    get_mean_state,
    predict_belief,
)
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


def test_belief_initialization():
    """Test that belief is correctly initialized from camera observation."""
    env = TabletopPickEnv(gui=False, num_objects=3)
    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert len(env.belief.particles) == 100
    assert np.isclose(env.belief.weights.sum(), 1.0)
    assert len(env.belief.known_objects) + len(env.belief.unknown_objects) == 3

    env.close()


def test_belief_predict():
    """Test belief prediction step."""
    env = TabletopPickEnv(gui=False, num_objects=3)
    _, _ = env.reset(seed=42)

    initial_belief = env.belief
    assert initial_belief is not None

    action = np.array([0.1, 0, 0, 0, 0, 0, 0])
    predicted_belief = predict_belief(
        initial_belief,
        action,
        np.array(env.robot.joint_lower_limits[:7]),
        np.array(env.robot.joint_upper_limits[:7]),
        noise_std=0.01,
    )

    assert len(predicted_belief.particles) == len(initial_belief.particles)
    assert np.isclose(predicted_belief.weights.sum(), 1.0)

    env.close()


def test_belief_update():
    """Test belief update step."""
    env = TabletopPickEnv(gui=False, num_objects=3)
    _, _ = env.reset(seed=42)

    initial_belief = env.belief
    assert initial_belief is not None

    action = np.array([0.1, 0, 0, 0, 0, 0, 0])
    env.step(action)

    updated_belief = env.belief
    assert updated_belief is not None
    assert np.isclose(updated_belief.weights.sum(), 1.0)

    env.close()


def test_get_mean_state():
    """Test extracting mean state from belief."""
    env = TabletopPickEnv(gui=False, num_objects=3)
    _, _ = env.reset(seed=42)

    belief = env.belief
    assert belief is not None

    mean_state = get_mean_state(belief)

    assert isinstance(mean_state, TabletopState)
    assert len(mean_state.joint_positions) == 9
    assert 0.0 <= mean_state.gripper_open <= 1.0

    env.close()


def test_belief_over_multiple_steps():
    """Test belief tracking over multiple steps."""
    env = TabletopPickEnv(gui=False, num_objects=5)
    _, _ = env.reset(seed=42)

    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)
        assert env.belief is not None
        assert np.isclose(env.belief.weights.sum(), 1.0)

    env.close()
