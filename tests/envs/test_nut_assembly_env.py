"""Tests for NutAssemblyEnv."""

from __future__ import annotations

from residual_controllers.envs.nut_assembly_env import NutAssemblyEnv


def test_basic_reset():
    """Verify NutAssemblyEnv resets correctly."""
    env = NutAssemblyEnv(gui=False)

    obs, info = env.reset(seed=42)

    assert set(obs.keys()) == {
        "joint_positions",
        "camera_pose",
        "object_poses",
        "held_object_idx",
        "grasp_transform",
    }
    assert obs["joint_positions"].shape == (9,)
    assert obs["camera_pose"].shape == (7,)
    assert obs["object_poses"].shape == (10, 7)
    assert info["target_object_label"] == "NUT"

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        assert not terminated
        assert reward == 0.0

    env.close()
