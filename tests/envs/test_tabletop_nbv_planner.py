"""Test NBV planner for information gathering."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import multiply_poses
from pybullet_helpers.gui import visualize_pose

from residual_controllers.envs.tabletop_pybullet import TabletopPickEnv
from residual_controllers.information_gathering import (
    NBVPlanner,
    compute_object_target_center,
    compute_region_centroid,
    filter_reachable_viewpoints,
    sample_hemisphere_viewpoints,
)
from residual_controllers.utils import invert_pose


@pytest.mark.skip(reason="Requires GUI interaction")
def test_nbv_planner_basic():
    """Test that NBV planner returns a valid viewpoint."""
    env = TabletopPickEnv(gui=True, num_objects=3, occlusion_prob=1.0)
    _, _ = env.reset(seed=42)

    assert env.belief is not None
    assert env.belief.visibility_grid is not None

    for _ in range(1):
        _, _, _, _, _ = env.step(np.zeros(8))

    grid_debug_ids = env.belief.visibility_grid.visualize(
        env.physics_client_id, z_range=(0.0, 0.1)
    )

    target_center = compute_region_centroid(
        env.belief.visibility_grid, z_range=(0.0, 0.1)
    )
    print(f"\nTarget center (uncertain region centroid): {target_center}")

    p.addUserDebugPoints(
        [target_center.tolist()],
        [[1, 1, 0]],
        pointSize=20,
        physicsClientId=env.physics_client_id,
    )

    candidate_poses = sample_hemisphere_viewpoints(
        center=target_center,
        radius_range=(0.25, 0.35),
        n_samples=50,
        elevation_range=(0.6, 1.3),
        seed=42,
    )

    print(f"Sampled {len(candidate_poses)} viewpoints")

    all_positions = [pose.position for pose in candidate_poses]
    p.addUserDebugPoints(
        all_positions,
        [[1, 0, 0]] * len(all_positions),
        pointSize=8,
        physicsClientId=env.physics_client_id,
    )

    reachable = filter_reachable_viewpoints(
        candidate_poses, env.robot, env.camera_to_ee_transform
    )
    print(f"Reachable viewpoints: {len(reachable)}/{len(candidate_poses)}")

    if len(reachable) > 0:
        reachable_positions = [pose.position for pose, _ in reachable]
        p.addUserDebugPoints(
            reachable_positions,
            [[0, 1, 0]] * len(reachable_positions),
            pointSize=12,
            physicsClientId=env.physics_client_id,
        )

    planner = NBVPlanner(
        robot=env.robot,
        camera_intrinsics=env.camera_intrinsics,
        camera_to_ee_transform=env.camera_to_ee_transform,
        n_viewpoint_samples=50,
        radius_range=(0.25, 0.35),
        elevation_range=(0.6, 1.3),
    )

    best_viewpoint = planner.plan(
        occupancy_grid=env.belief.visibility_grid,
        z_range=(0.0, 0.1),
        seed=42,
    )

    print("\nNBV Planner Result:")
    if best_viewpoint is not None:
        print(f"  Position: {best_viewpoint.pose.position}")
        print(f"  Expected IG: {best_viewpoint.expected_ig}")
        print(f"  Joint config: {best_viewpoint.joint_config[:3]}...")

        p.addUserDebugPoints(
            [best_viewpoint.pose.position],
            [[0, 0, 1]],
            pointSize=20,
            physicsClientId=env.physics_client_id,
        )

        input("Press Enter to move robot to NBV...")

        current_joints = np.array(env.robot.get_joint_positions())
        target_joints = np.array(best_viewpoint.joint_config)
        joint_delta = target_joints[:7] - current_joints[:7]
        action = np.concatenate([joint_delta, [0.0]])
        action = np.clip(action, env.action_space.low, env.action_space.high)

        _, _, _, _, _ = env.step(action)

        print("\nAfter info-gathering step:")
        print(f"  Known objects: {env.belief.known_objects}")
        print(f"  Unknown objects: {env.belief.unknown_objects}")

        env.belief.visibility_grid.clear_visualization(
            grid_debug_ids, env.physics_client_id
        )
        grid_debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.1)
        )
    else:
        print("  No valid viewpoint found")

    input("Press Enter to close...")

    env.belief.visibility_grid.clear_visualization(
        grid_debug_ids, env.physics_client_id
    )
    env.close()


@pytest.mark.skip(reason="Requires GUI interaction")
def test_nbv_planner_for_target_object():
    """Test NBV planning to observe the target object specifically."""
    env = TabletopPickEnv(gui=True, num_objects=3, occlusion_prob=1.0)
    _, info = env.reset(seed=42)

    target_object_id = info["target_object_id"]

    assert env.belief is not None
    assert env.scene is not None

    for _ in range(1):
        _, _, _, _, _ = env.step(np.zeros(8))

    print(f"\n{'='*60}")
    print("NBV Planning for Target Object")
    print(f"{'='*60}")
    print(f"Target object ID: {target_object_id}")
    print(f"Known objects: {env.belief.known_objects}")
    print(f"Unknown objects: {env.belief.unknown_objects}")
    print(f"Occluded objects: {env.belief.occluded_objects}")

    grid_debug_ids = env.belief.visibility_grid.visualize(
        env.physics_client_id, z_range=(0.0, 0.1)
    )

    planner = NBVPlanner(
        robot=env.robot,
        camera_intrinsics=env.camera_intrinsics,
        camera_to_ee_transform=env.camera_to_ee_transform,
        n_viewpoint_samples=50,
        radius_range=(0.05, 0.2),
        elevation_range=(0.6, 0.8),
    )

    max_steps = 5
    target_debug_id = None
    best_debug_id = None
    pose_debug_ids: set[int] = set()
    for step in range(max_steps):
        target_center = compute_object_target_center(
            env.belief, target_object_id, z_range=(0.0, 0.1)
        )

        if target_center is not None:
            if target_debug_id is not None:
                p.removeUserDebugItem(
                    target_debug_id, physicsClientId=env.physics_client_id
                )
            target_debug_id = p.addUserDebugPoints(
                [target_center.tolist()],
                [[1, 1, 0]],
                pointSize=20,
                physicsClientId=env.physics_client_id,
            )

        best_viewpoint = planner.plan_for_object(
            belief=env.belief,
            object_id=target_object_id,
            z_range=(0.0, 0.1),
            seed=42 + step,
        )

        print(f"\nNBV Planner Result (step {step + 1}/{max_steps}):")
        if best_viewpoint is None:
            print("  No valid viewpoint found")
            break

        print(f"  Position: {best_viewpoint.pose.position}")
        print(f"  Expected IG: {best_viewpoint.expected_ig}")

        for debug_id in pose_debug_ids:
            p.removeUserDebugItem(debug_id, physicsClientId=env.physics_client_id)
        pose_debug_ids.clear()

        if env.camera_to_ee_transform is not None:
            ee_pose = multiply_poses(
                best_viewpoint.pose, invert_pose(env.camera_to_ee_transform)
            )
            pose_debug_ids.update(
                visualize_pose(ee_pose, env.physics_client_id, axis_length=0.08)
            )
        pose_debug_ids.update(
            visualize_pose(best_viewpoint.pose, env.physics_client_id, axis_length=0.08)
        )

        if target_center is not None:
            cam_pos = np.array(best_viewpoint.pose.position, dtype=np.float32)
            cam_rot = np.array(
                p.getMatrixFromQuaternion(best_viewpoint.pose.orientation),
                dtype=np.float32,
            ).reshape((3, 3))
            cam_forward = cam_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            to_target = target_center - cam_pos
            to_target_norm = np.linalg.norm(to_target) + 1e-8
            cam_forward_norm = np.linalg.norm(cam_forward) + 1e-8
            dot = float(
                np.dot(cam_forward, to_target) / (cam_forward_norm * to_target_norm)
            )
            print(f"  Camera forward dot target: {dot:.3f}")

        if best_debug_id is not None:
            p.removeUserDebugItem(best_debug_id, physicsClientId=env.physics_client_id)
        best_debug_id = p.addUserDebugPoints(
            [best_viewpoint.pose.position],
            [[0, 0, 1]],
            pointSize=20,
            physicsClientId=env.physics_client_id,
        )

        input("Press Enter to move robot to NBV...")

        target_joints = np.array(best_viewpoint.joint_config)
        for _ in range(20):
            current_joints = np.array(env.robot.get_joint_positions())
            joint_delta = target_joints[:7] - current_joints[:7]
            if np.linalg.norm(joint_delta) < 1e-3:
                break
            action = np.concatenate([joint_delta, [0.0]])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, _, _, _ = env.step(action)

        print("\nAfter info-gathering step:")
        print(f"  Known objects: {env.belief.known_objects}")
        print(f"  Unknown objects: {env.belief.unknown_objects}")
        print(f"  Target in known: {target_object_id in env.belief.known_objects}")

        env.belief.visibility_grid.clear_visualization(
            grid_debug_ids, env.physics_client_id
        )
        grid_debug_ids = env.belief.visibility_grid.visualize(
            env.physics_client_id, z_range=(0.0, 0.1)
        )

        # if target_object_id in env.belief.known_objects:
        #     break

    input("Press Enter to close...")

    env.belief.visibility_grid.clear_visualization(
        grid_debug_ids, env.physics_client_id
    )
    env.close()
