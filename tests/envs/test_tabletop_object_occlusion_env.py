"""Tests for TabletopObjectOcclusionEnv."""

from __future__ import annotations

import numpy as np

from residual_controllers.envs.tabletop_object_occlusion import (
    TabletopObjectOcclusionEnv,
)


def test_basic_reset():
    """Verify TabletopObjectOcclusionEnv resets."""
    env = TabletopObjectOcclusionEnv(gui=False)

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
    assert info["target_object_label"] == "MILK"

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        assert not terminated
        assert reward == 0.0

    env.close()


def test_milk_initially_occluded():
    """Verify that the milk carton is not visible in the initial wrist camera
    image."""
    env = TabletopObjectOcclusionEnv(gui=False)
    env.reset(seed=42)

    assert env.scene is not None
    camera_image = env.get_camera_image()
    milk_id = env.scene.object_ids[env.scene.milk_carton_idx]

    milk_visible = np.any(camera_image.segmentation == milk_id)
    assert not milk_visible, "Milk carton should be fully occluded at reset"

    env.close()
