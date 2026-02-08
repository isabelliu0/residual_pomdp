"""Integration test for end-to-end belief tracking in TabletopPickEnv."""

import numpy as np

from residual_controllers.beliefs import get_mean_state
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


def test_end_to_end_belief_tracking():
    """Test full belief tracking pipeline over episode."""
    env = TabletopPickEnv(gui=False, num_objects=5, occlusion_prob=0.7)

    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert len(env.belief.particles) > 0
    assert np.isclose(env.belief.weights.sum(), 1.0)

    for _ in range(20):
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)

        assert env.belief is not None
        assert np.isclose(env.belief.weights.sum(), 1.0)
        assert len(env.belief.particles) > 0

    mean_state = get_mean_state(env.belief)
    assert mean_state is not None
    assert len(mean_state.joint_positions) == 9

    target_id = env.scene.object_ids[env.scene.target_idx]
    assert target_id in env.scene.object_ids

    env.close()


def test_belief_tracks_known_unknown_objects():
    """Test that belief correctly tracks known vs unknown objects."""
    env = TabletopPickEnv(gui=False, num_objects=5, occlusion_prob=0.9)

    _, _ = env.reset(seed=123)

    assert env.belief is not None

    total_objects = len(env.belief.known_objects) + len(env.belief.unknown_objects)
    assert total_objects == env.num_objects

    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)

        assert env.belief is not None
        total = len(env.belief.known_objects) + len(env.belief.unknown_objects)
        assert total == env.num_objects

    env.close()


def test_visibility_grid_updates():
    """Test that visibility grid is updated during belief tracking."""
    env = TabletopPickEnv(gui=False, num_objects=3)

    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert env.belief.visibility_grid is not None

    initial_grid = env.belief.visibility_grid.grid.copy()

    for _ in range(5):
        action = np.array([0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, _, _ = env.step(action)

    final_grid = env.belief.visibility_grid.grid

    assert not np.allclose(initial_grid, final_grid)

    env.close()
