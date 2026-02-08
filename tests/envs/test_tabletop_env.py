"""Test for the TabletopPickEnv environment."""

import time

from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


def test_basic_env():
    """Basic test."""
    print("Creating environment...")
    env = TabletopPickEnv(gui=True, num_objects=5)

    print("Resetting environment...")
    obs, info = env.reset(seed=42)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Joint positions shape: {obs['joint_positions'].shape}")
    print(f"Camera pose shape: {obs['camera_pose'].shape}")
    print(f"Target object ID: {info['target_object_id']}")

    print("\nCapturing camera image...")
    camera_image = env.get_camera_image()
    print(f"RGB shape: {camera_image.rgb.shape}")
    print(f"Depth shape: {camera_image.depth.shape}")

    print("\nRunning 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        time.sleep(0.1)
        print(f"Step {i+1}: reward={reward}, terminated={terminated}")

    print("\nTest completed successfully!")
    env.close()
