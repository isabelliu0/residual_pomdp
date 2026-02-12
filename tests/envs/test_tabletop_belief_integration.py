"""Integration test for end-to-end belief tracking in TabletopPickEnv."""

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import run_smooth_motion_planning_to_pose

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


def test_nominal_policy_reach_target():
    """Test nominal policy that reaches toward target with visibility grid
    monitoring."""
    env = TabletopPickEnv(gui=False, num_objects=3, occlusion_prob=0.5)
    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert env.scene is not None

    target_id = env.scene.object_ids[env.scene.target_idx]
    target_pose_ground_truth = p.getBasePositionAndOrientation(
        target_id, physicsClientId=env.physics_client_id
    )
    target_pos = np.array(target_pose_ground_truth[0])

    print(f"\n{'='*60}")
    print(f"Target object ID: {target_id}")
    print(f"Target position (ground truth): {target_pos}")
    print(f"Initial known objects: {len(env.belief.known_objects)}/{env.num_objects}")
    print(f"{'='*60}\n")

    _print_visibility_stats(env, step=0)

    robot_orientation = env.robot.get_end_effector_pose().orientation
    approach_position = target_pos + np.array([0.0, 0.0, 0.15])
    approach_pose = Pose(tuple(approach_position), robot_orientation)

    print(f"\nPlanning motion to approach pose: {approach_position}")
    plan = run_smooth_motion_planning_to_pose(
        approach_pose,
        env.robot,
        collision_ids=[],  # [env.scene.table_id],
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=42,
        max_time=1.0,
    )

    print(f"Executing plan with {len(plan)} waypoints")
    current_joints = np.array(env.robot.get_joint_positions())

    for i, target_joints in enumerate(plan):
        joint_delta = np.subtract(target_joints[:7], current_joints[:7])
        action = joint_delta[:7]
        action = np.clip(action, env.action_space.low, env.action_space.high)

        _, _, _, _, _ = env.step(action)
        current_joints = np.array(env.robot.get_joint_positions())

        if i % 5 == 0:
            _print_visibility_stats(env, step=i + 1)

    print(f"\n{'='*60}")
    print("FINAL STATE")
    print(f"{'='*60}")
    _print_visibility_stats(env, step="FINAL")

    final_ee_pos = np.array(env.robot.get_end_effector_pose().position)
    distance_to_target = np.linalg.norm(final_ee_pos - target_pos)
    print(f"Distance to target: {distance_to_target:.3f}m")
    print(f"Known objects: {len(env.belief.known_objects)}/{env.num_objects}")
    print(f"{'='*60}\n")

    assert env.belief is not None
    assert distance_to_target < 0.3

    env.close()


def _print_visibility_stats(env: TabletopPickEnv, step: int | str) -> None:
    """Helper to print visibility grid statistics."""
    if env.belief is None or env.belief.visibility_grid is None:
        return

    grid = env.belief.visibility_grid.grid
    probs = env.belief.visibility_grid.get_occupancy_probabilities()

    free_voxels = np.sum(probs < 0.3)
    uncertain_voxels = np.sum((probs >= 0.3) & (probs <= 0.7))
    occupied_voxels = np.sum(probs > 0.7)
    total_voxels = grid.size

    mean_log_odds = np.mean(grid)
    min_log_odds = np.min(grid)
    max_log_odds = np.max(grid)

    print(f"Step {step}:")
    print(
        f"  Free voxels (p<0.3):      {free_voxels:5d} ({100*free_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Uncertain (0.3≤p≤0.7):    {uncertain_voxels:5d} ({100*uncertain_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Occupied voxels (p>0.7):  {occupied_voxels:5d} ({100*occupied_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Log-odds: mean={mean_log_odds:+.2f}, min={min_log_odds:+.2f}, max={max_log_odds:+.2f}"  # pylint: disable=line-too-long
    )
    print(
        f"  Known objects: {len(env.belief.known_objects)}, Unknown: {len(env.belief.unknown_objects)}"  # pylint: disable=line-too-long
    )
    print()
