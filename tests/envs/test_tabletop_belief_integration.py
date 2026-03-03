"""Integration test for end-to-end belief tracking in TabletopPickEnv."""

import os
import time

import imageio
import numpy as np
import pybullet as p
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import run_smooth_motion_planning_to_pose

from residual_controllers.beliefs import (
    LogOddsOccupancyGrid,
    compute_belief_diagnostics,
)
from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv


def test_nominal_policy_reach_target():
    """Test nominal policy that reaches toward target with visibility grid
    monitoring and visualization."""
    base_env = TabletopPickEnv(
        gui=False, num_objects=3, occlusion_prob=0.5, render_mode="rgb_array"
    )
    env = RecordVideo(base_env, "videos/belief-integration-test")
    sim = TabletopPickEnv(gui=False, num_objects=3, occlusion_prob=0.5)

    _, _ = env.reset(seed=42)
    _, _ = sim.reset(seed=42)

    assert base_env.belief is not None
    assert base_env.scene is not None

    target_id = base_env.scene.object_ids[base_env.scene.target_idx]
    target_pose_ground_truth = p.getBasePositionAndOrientation(
        target_id, physicsClientId=base_env.physics_client_id
    )
    target_pos = np.array(target_pose_ground_truth[0])

    print(f"\n{'='*60}")
    print(f"Target object ID: {target_id}")
    print(f"Target position (ground truth): {target_pos}")
    print(
        f"Initial known objects: {len(base_env.belief.known_objects)}/{base_env.num_objects}"  # pylint: disable=line-too-long
    )
    print(f"{'='*60}\n")

    _print_visibility_stats(base_env, step=0)

    debug_ids: list[int] = []
    # occupied_debug_ids: list[int] = []
    if base_env.belief.visibility_grid is not None:
        debug_ids = base_env.belief.visibility_grid.visualize(
            base_env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = base_env.belief.visibility_grid.visualize_occupied(
        #     base_env.physics_client_id, z_range=(0.0, 0.15)
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
    current_joints = np.array(base_env.robot.get_joint_positions())

    for i, target_joints in enumerate(plan):
        joint_delta = np.subtract(target_joints[:7], current_joints[:7])
        action = np.concatenate([joint_delta[:7], [0.0]])
        action = np.clip(action, base_env.action_space.low, base_env.action_space.high)

        _, _, _, _, _ = env.step(action)
        current_joints = np.array(base_env.robot.get_joint_positions())

        LogOddsOccupancyGrid.clear_visualization(debug_ids, base_env.physics_client_id)
        # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, base_env.physics_client_id)  # pylint: disable=line-too-long
        debug_ids = base_env.belief.visibility_grid.visualize(
            base_env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = base_env.belief.visibility_grid.visualize_occupied(
        #     base_env.physics_client_id, z_range=(0.0, 0.15)
        # )
        time.sleep(0.5)

        if i % 5 == 0:
            _print_visibility_stats(base_env, step=i + 1)

    print(f"\n{'='*60}")
    print("FINAL STATE")
    print(f"{'='*60}")
    _print_visibility_stats(base_env, step="FINAL")

    final_ee_pos = np.array(base_env.robot.get_end_effector_pose().position)
    distance_to_target = np.linalg.norm(final_ee_pos - target_pos)
    print(f"Distance to target: {distance_to_target:.3f}m")
    print(f"Known objects: {len(base_env.belief.known_objects)}/{base_env.num_objects}")
    print(f"{'='*60}\n")

    assert base_env.belief is not None
    assert distance_to_target < 0.3

    LogOddsOccupancyGrid.clear_visualization(debug_ids, base_env.physics_client_id)
    # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, base_env.physics_client_id)  # pylint: disable=line-too-long

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
    env = TabletopPickEnv(gui=False, num_objects=5, occlusion_prob=0.7)
    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert env.scene is not None

    os.makedirs("videos", exist_ok=True)
    external_writer = imageio.get_writer(
        "videos/belief-integration-test/particle_filter_external.mp4",
        fps=10,
        codec="libx264",
    )
    wrist_writer = imageio.get_writer(
        "videos/belief-integration-test/particle_filter_wrist.mp4",
        fps=10,
        codec="libx264",
    )

    def capture_frames():
        external_frame = capture_image(
            env.physics_client_id,
            camera_distance=1.5,
            camera_yaw=50,
            camera_pitch=-35,
            camera_target=(0.5, 0.0, 0.0),
            image_width=640,
            image_height=480,
        )
        external_writer.append_data(external_frame)
        wrist_frame = env.get_camera_image().rgb
        wrist_writer.append_data(wrist_frame)

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

    capture_frames()

    scan_actions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    resampling_count = 0

    for step, action in enumerate(scan_actions):
        camera_image = env.get_camera_image()

        diagnostics = compute_belief_diagnostics(
            env.belief,
            camera_image,
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

        if diagnostics.resampled:
            resampling_count += 1

        _, _, _, _, _ = env.step(action)
        capture_frames()

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

    external_writer.close()
    wrist_writer.close()
    env.close()


def test_table_surface_unseen_voxels():
    """Print unseen voxels near table surface after reset and where unknown
    object particles fall."""
    env = TabletopPickEnv(gui=False, num_objects=5, occlusion_prob=0.7)
    env.reset(seed=42)

    assert env.belief is not None
    assert env.belief.visibility_grid is not None
    assert env.scene is not None

    debug_ids: list[int] = []
    # occupied_debug_ids: list[int] = []
    if env.belief.visibility_grid is not None:
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = env.belief.visibility_grid.visualize_occupied(
        #     env.physics_client_id, z_range=(0.0, 0.15)
        # )

    noop = np.zeros(8, dtype=np.float32)
    for _ in range(1):
        env.step(noop)
        LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)
        # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, env.physics_client_id)  # pylint: disable=line-too-long
        debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.15)
        )
        # occupied_debug_ids = env.belief.visibility_grid.visualize_occupied(
        #     env.physics_client_id, z_range=(0.0, 0.15)
        # )
        time.sleep(0.5)

    grid = env.belief.visibility_grid
    table_z_min, table_z_max = 0.0, 0.05

    unseen = grid.get_unobserved_voxels()
    unseen_table = [(x, y, z) for x, y, z in unseen if table_z_min <= z <= table_z_max]
    free_table = [
        (x, y, z)
        for x, y, z in grid.get_free_voxels()
        if table_z_min <= z <= table_z_max
    ]

    print(f"\n{'='*60}")
    print(f"TABLE SURFACE UNSEEN VOXELS (z in [{table_z_min}, {table_z_max}])")
    print(f"  Unseen: {len(unseen_table)}   Free: {len(free_table)}")

    if unseen_table:
        xs = [v[0] for v in unseen_table]
        ys = [v[1] for v in unseen_table]
        print(
            f"  x range: [{min(xs):.3f}, {max(xs):.3f}]   y range: [{min(ys):.3f}, {max(ys):.3f}]"  # pylint: disable=line-too-long
        )

        x_bins = np.linspace(0.0, 1.0, 9)
        y_bins = np.linspace(-0.5, 0.5, 7)
        counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)
        for x, y, _ in unseen_table:
            xi = int(np.clip(np.searchsorted(x_bins, x) - 1, 0, counts.shape[0] - 1))
            yi = int(np.clip(np.searchsorted(y_bins, y) - 1, 0, counts.shape[1] - 1))
            counts[xi, yi] += 1
        print("\n  2D density of unseen voxels (x=rows, y=cols):")
        print(
            "  x\\y  "
            + "  ".join(
                f"{(y_bins[i]+y_bins[i+1])/2:+.2f}" for i in range(len(y_bins) - 1)
            )
        )
        for i in range(counts.shape[0]):
            row = "  ".join(f"{counts[i, j]:4d}" for j in range(counts.shape[1]))
            print(f"  {(x_bins[i]+x_bins[i+1])/2:.2f}  {row}")

    print(
        f"\n  Known: {env.belief.known_objects}   Unknown: {env.belief.unknown_objects}"
    )

    all_particle_poses: dict[int, list] = {}
    for particle in env.belief.particles:
        for obj_id in env.scene.object_ids:
            if obj_id in env.belief.unknown_objects and obj_id in particle.object_poses:
                all_particle_poses.setdefault(obj_id, []).append(
                    particle.object_poses[obj_id]
                )

    for obj_id, poses in all_particle_poses.items():
        n_in_unseen = sum(
            1
            for pos in poses
            if not grid.is_free((pos[0], pos[1], pos[2]))
            and not grid.is_occupied((pos[0], pos[1], pos[2]))
        )
        print(
            f"  Unknown obj {obj_id}: {n_in_unseen}/{len(poses)} particles in unseen region"  # pylint: disable=line-too-long
        )

    print(f"{'='*60}\n")
    assert len(unseen_table) > 0, "Expected unseen voxels at table surface after reset"

    LogOddsOccupancyGrid.clear_visualization(debug_ids, env.physics_client_id)
    # LogOddsOccupancyGrid.clear_visualization(occupied_debug_ids, env.physics_client_id)  # pylint: disable=line-too-long
    env.close()


def test_unknown_target_particle_distribution():
    """Print particle distribution for the target object after reset."""
    env = TabletopPickEnv(gui=False, num_objects=5, occlusion_prob=0.7)
    _, info = env.reset(seed=42)

    assert env.belief is not None
    assert env.belief.visibility_grid is not None
    assert env.scene is not None

    target_id = info.get("target_object_id")
    assert target_id is not None

    gt_pos = np.array(
        p.getBasePositionAndOrientation(
            target_id, physicsClientId=env.physics_client_id
        )[0]
    )

    status = (
        "known"
        if target_id in env.belief.known_objects
        else "occluded" if target_id in env.belief.occluded_objects else "unknown"
    )

    particle_xys = np.array(
        [
            [particle.object_poses[target_id][0], particle.object_poses[target_id][1]]
            for particle in env.belief.particles
            if target_id in particle.object_poses
        ]
    )

    print(f"\n{'='*60}")
    print("TARGET OBJECT PARTICLE DISTRIBUTION")
    print(
        f"  target_id={target_id}  status={status}  confidence={env.belief.object_confidence.get(target_id, 0.0):.3f}"  # pylint: disable=line-too-long
    )
    print(f"  ground truth (x,y): ({gt_pos[0]:.3f}, {gt_pos[1]:.3f})")
    print(f"  n particles: {len(particle_xys)}")

    if len(particle_xys) > 0:
        mean_xy = particle_xys.mean(axis=0)
        std_xy = particle_xys.std(axis=0)
        print(f"  mean (x,y): ({mean_xy[0]:.3f}, {mean_xy[1]:.3f})")
        print(f"  std  (x,y): ({std_xy[0]:.3f}, {std_xy[1]:.3f})")
        print(f"  mean error from GT: {np.linalg.norm(mean_xy - gt_pos[:2]):.3f} m")

        x_bins = np.linspace(0.0, 1.0, 9)
        y_bins = np.linspace(-0.5, 0.5, 7)
        counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)
        for x, y in particle_xys:
            xi = int(np.clip(np.searchsorted(x_bins, x) - 1, 0, counts.shape[0] - 1))
            yi = int(np.clip(np.searchsorted(y_bins, y) - 1, 0, counts.shape[1] - 1))
            counts[xi, yi] += 1
        print("\n  2D particle density (x=rows, y=cols):")
        print(
            "  x\\y  "
            + "  ".join(
                f"{(y_bins[i]+y_bins[i+1])/2:+.2f}" for i in range(len(y_bins) - 1)
            )
        )
        for i in range(counts.shape[0]):
            row = "  ".join(f"{counts[i, j]:4d}" for j in range(counts.shape[1]))
            print(f"  {(x_bins[i]+x_bins[i+1])/2:.2f}  {row}")

        grid = env.belief.visibility_grid
        n_in_unseen = sum(
            1
            for x, y in particle_xys
            if not grid.is_free((float(x), float(y), gt_pos[2]))
            and not grid.is_occupied((float(x), float(y), gt_pos[2]))
        )
        print(f"\n  particles in unseen region: {n_in_unseen}/{len(particle_xys)}")

    print(f"{'='*60}\n")
