"""Integration test for end-to-end belief tracking in TabletopPickEnv."""

import time

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import run_smooth_motion_planning_to_pose

from residual_controllers.beliefs import (
    LogOddsOccupancyGrid,
    compute_belief_diagnostics,
    get_mean_state,
)
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
    monitoring and visualization."""
    env = TabletopPickEnv(gui=False, num_objects=3, occlusion_prob=0.5)
    sim = TabletopPickEnv(gui=False, num_objects=3, occlusion_prob=0.5)

    _, _ = env.reset(seed=42)
    _, _ = sim.reset(seed=42)

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

    debug_ids: list[int] = []
    # occupied_debug_ids: list[int] = []
    if env.belief.visibility_grid is not None:
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = env.belief.visibility_grid.visualize_occupied(
        #     env.physics_client_id, z_range=(0.0, 0.15)
        # )

    robot_orientation = sim.robot.get_end_effector_pose().orientation
    approach_position = target_pos + np.array([0.0, 0.0, 0.15])
    approach_pose = Pose(tuple(approach_position), robot_orientation)

    print(f"\nPlanning motion to approach pose: {approach_position}")
    plan = run_smooth_motion_planning_to_pose(
        approach_pose,
        sim.robot,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=42,
        max_time=1.0,
    )

    if plan is None:
        print("Motion planning failed")
        env.close()
        sim.close()
        return

    print(f"Executing plan with {len(plan)} waypoints")
    current_joints = np.array(env.robot.get_joint_positions())

    for i, target_joints in enumerate(plan):
        joint_delta = np.subtract(target_joints[:7], current_joints[:7])
        action = joint_delta[:7]
        action = np.clip(action, env.action_space.low, env.action_space.high)

        _, _, _, _, _ = env.step(action)
        current_joints = np.array(env.robot.get_joint_positions())

        LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)
        # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, env.physics_client_id)   # pylint: disable=line-too-long
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = env.belief.visibility_grid.visualize_occupied(
        #     env.physics_client_id, z_range=(0.0, 0.15)
        # )
        time.sleep(0.5)

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

    LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)
    # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, env.physics_client_id)

    env.close()
    sim.close()


def _print_visibility_stats(env: TabletopPickEnv, step: int | str) -> None:
    """Helper to print visibility grid statistics."""
    if env.belief is None or env.belief.visibility_grid is None:
        return

    grid = env.belief.visibility_grid.grid
    probs = env.belief.visibility_grid.get_occupancy_probabilities()

    free_max = env.belief.visibility_grid.thresholds.free_max
    occupied_min = env.belief.visibility_grid.thresholds.occupied_min
    free_voxels = np.sum(probs < free_max)
    uncertain_voxels = np.sum((probs >= free_max) & (probs <= occupied_min))
    occupied_voxels = np.sum(probs > occupied_min)
    total_voxels = grid.size

    mean_log_odds = np.mean(grid)
    min_log_odds = np.min(grid)
    max_log_odds = np.max(grid)

    print(f"Step {step}:")
    print(
        f"  Free voxels (p<{free_max:.2f}):      {free_voxels:5d} ({100*free_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Uncertain ({free_max:.2f}≤p≤{occupied_min:.2f}):    {uncertain_voxels:5d} ({100*uncertain_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Occupied voxels (p>{occupied_min:.2f}):  {occupied_voxels:5d} ({100*occupied_voxels/total_voxels:5.1f}%)"  # pylint: disable=line-too-long
    )
    print(
        f"  Log-odds: mean={mean_log_odds:+.2f}, min={min_log_odds:+.2f}, max={max_log_odds:+.2f}"  # pylint: disable=line-too-long
    )
    print(
        f"  Known objects: {len(env.belief.known_objects)}, Unknown: {len(env.belief.unknown_objects)}"  # pylint: disable=line-too-long
    )
    print()


def test_particle_filter_diagnostics():
    """Test particle filter with camera tilt sequence to reveal objects."""
    env = TabletopPickEnv(gui=False, num_objects=3, occlusion_prob=0.3)
    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert env.scene is not None

    print(f"\n{'='*70}")
    print("PARTICLE FILTER DIAGNOSTICS TEST")
    print(f"{'='*70}")
    print(f"Num particles: {len(env.belief.particles)}")
    print(f"Num objects: {env.num_objects}")
    print(f"Known objects: {env.belief.known_objects}")
    print(f"Unknown objects: {env.belief.unknown_objects}")

    debug_ids: list[int] = []
    if env.belief.visibility_grid is not None:
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )

    scan_actions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    resampling_count = 0

    for step, action in enumerate(scan_actions):
        camera_image = env.get_camera_image()
        camera_pose = env.get_camera_pose_se3()

        diagnostics = compute_belief_diagnostics(
            env.belief,
            camera_image,
            camera_pose,
            env.camera_intrinsics,
            env.scene.object_ids,
            env.physics_client_id,
        )

        print(f"\n--- Step {step} ---")
        print(
            f"  n_eff: {diagnostics.n_eff_before:.1f} -> {diagnostics.n_eff_after:.1f}"
        )
        print(
            f"  entropy: {diagnostics.weight_entropy_before:.2f} -> {diagnostics.weight_entropy_after:.2f}"  # pylint: disable=line-too-long
        )
        print(f"  ego_confidence: {diagnostics.ego_pose_confidence:.3f}")
        print(f"  will_resample: {diagnostics.resampled}")
        for obj_id in env.scene.object_ids:
            if obj_id in env.belief.known_objects:
                status = "known"
            elif obj_id in env.belief.occluded_objects:
                status = "occluded"
            else:
                status = "unknown"
            confidence = env.belief.object_confidence.get(obj_id, 0.0)
            print(f"  Object {obj_id} status: {status}, confidence={confidence:.3f}")

        for obj_id, stats in diagnostics.object_stats.items():
            print(f"  Object {obj_id}:")
            print(
                f"   dist: mean={stats.mean_distance:.4f}, std={stats.std_distance:.4f},"
                f"[{stats.min_distance:.4f}, {stats.max_distance:.4f}]"
            )
            print(
                f"   likelihood: mean={stats.mean_likelihood:.4f},"
                f"range=[{stats.min_likelihood:.4f}, {stats.max_likelihood:.4f}]"
            )
            print(
                f"   no poses: {stats.particles_with_none_pose}/{stats.total_particles}"
            )

        if diagnostics.resampled:
            resampling_count += 1

        _, _, _, _, _ = env.step(action)

        LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )

    print(f"\n{'='*70}")
    print(
        f"SUMMARY: Resampled {resampling_count}/{len(scan_actions)} steps "
        f"({100*resampling_count/len(scan_actions):.1f}%)"
    )
    print(f"Final known objects: {env.belief.known_objects}")
    print(f"Final unknown objects: {env.belief.unknown_objects}")
    print(f"{'='*70}\n")

    LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)

    env.close()
