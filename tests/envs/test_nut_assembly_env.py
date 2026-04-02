"""Tests for NutAssemblyEnv."""

from __future__ import annotations

import os
import time

import imageio
import numpy as np
import pybullet as p
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses

from residual_controllers.beliefs.perception import (
    detect_objects_from_pointcloud_pca,
    detect_objects_from_segmentation,
    transform_point,
)
from residual_controllers.beliefs.structs import CameraIntrinsics, TabletopState
from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.envs.nut_assembly_env import (
    NutAssemblyEnv,
    NutAssemblyEnvState,
)
from residual_controllers.tamp.pddl_utils import build_object_interaction_graph


def _unproject_mask_points(
    segmentation: np.ndarray,
    depth: np.ndarray,
    obj_id: int,
    camera_pose: tuple[tuple[float, ...], tuple[float, ...]],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Unproject all segmentation mask pixels for obj_id to world-frame 3D
    points."""
    mask = segmentation == obj_id
    if not np.any(mask):
        return np.empty((0, 3))
    ys, xs = np.where(mask)
    points = []
    for y, x in zip(ys, xs):
        d = float(depth[y, x])
        if d <= 0:
            continue
        point_cam = intrinsics.unproject(float(x), float(y), d)
        point_world = transform_point(point_cam, camera_pose)
        points.append(point_world)
    return np.array(points) if points else np.empty((0, 3))


def _extract_top_face_points(
    points_world: np.ndarray,
    initial_top_fraction: float = 0.25,
    inlier_distance: float = 0.005,
) -> np.ndarray:
    """Find top-face points by fitting a plane to the highest-Z cluster."""
    if len(points_world) < 3:
        return np.empty((0, 3))

    z_threshold = np.percentile(points_world[:, 2], (1 - initial_top_fraction) * 100)
    candidates = points_world[points_world[:, 2] >= z_threshold]

    if len(candidates) < 3:
        return candidates

    centroid = candidates.mean(axis=0)
    _, _, vh = np.linalg.svd(candidates - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal

    distances = np.abs((points_world - centroid) @ normal)
    return points_world[distances <= inlier_distance]


_PEG_HEIGHT = 0.1


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


def test_nut_peg_surface_interaction():
    """Visual test: nut held near peg top slides along surface without penetrating."""
    env = NutAssemblyEnv(gui=False)
    env.reset(seed=42)

    robot = env.robot
    pcid = env.physics_client_id
    peg_cx, peg_cy = 0.5, 0.0
    peg_top_z = _PEG_HEIGHT

    _, _, ee_yaw = p.getEulerFromQuaternion(robot.get_end_effector_pose().orientation)
    down_quat = tuple(p.getQuaternionFromEuler([np.pi, 0, ee_yaw]))

    ik = p.calculateInverseKinematics(
        robot.robot_id,
        robot.end_effector_id,
        (peg_cx, peg_cy, peg_top_z + 0.1),
        targetOrientation=down_quat,
        maxNumIterations=1000,
        residualThreshold=1e-6,
        physicsClientId=pcid,
    )
    joints = list(ik[:7]) + list(robot.get_joint_positions()[7:])
    robot.set_joints(joints)
    ee_pose = robot.get_end_effector_pose()

    nut_pose = Pose(position=(peg_cx, peg_cy, peg_top_z + 0.1), orientation=down_quat)
    peg_pose = Pose(position=(peg_cx, peg_cy, peg_top_z / 2), orientation=(0, 0, 0, 1))
    grasp_tf = multiply_poses(ee_pose.invert(), nut_pose)

    env.set_state(
        NutAssemblyEnvState(
            robot_joints=tuple(joints),
            object_poses=(nut_pose, peg_pose),
            held_object_idx=0,
            grasp_transform=grasp_tf,
        )
    )

    def step_toward(target_pos: tuple, n_steps: int = 20) -> None:
        for _ in range(n_steps):
            ik_step = p.calculateInverseKinematics(
                robot.robot_id,
                robot.end_effector_id,
                target_pos,
                targetOrientation=down_quat,
                maxNumIterations=1000,
                residualThreshold=1e-6,
                physicsClientId=pcid,
            )
            curr = np.array(robot.get_joint_positions()[:7])
            delta = np.clip(np.array(ik_step[:7]) - curr, -0.01, 0.01)
            env.step(np.concatenate([delta, [0.0]]).astype(np.float32))

            contacts = p.getContactPoints(
                env._nut_id,  # pylint: disable=protected-access
                env._peg_id,  # pylint: disable=protected-access
                physicsClientId=pcid,
            )
            for c in contacts or []:
                assert (
                    float(c[8]) > -2e-3
                ), f"Nut penetrated peg by {-float(c[8])*1000:.2f} mm"

            time.sleep(0.04)

    rim_z = peg_top_z - 0.01
    offset = 0.025

    for angle in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        step_toward(
            (peg_cx + offset * np.cos(angle), peg_cy + offset * np.sin(angle), rim_z),
            n_steps=15,
        )

    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        step_toward(
            (peg_cx + offset * np.cos(angle), peg_cy + offset * np.sin(angle), rim_z),
            n_steps=4,
        )

    step_toward((peg_cx, peg_cy, rim_z), n_steps=20)

    env.close()


def test_orp_calibration():
    """ORP core assumption: lower peg belief sigma -> higher assembly success rate."""

    def _inject_peg_sigma(
        system: NutAssemblyTAMPSystem, peg_sigma: float, rng: np.random.Generator
    ) -> None:
        belief = system.env.belief
        if belief is None:
            return
        peg_gt = get_pose(
            system.env._peg_id, system.env.physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
        )
        nut_gt = get_pose(
            system.env._nut_id, system.env.physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
        )
        new_particles = []
        for particle in belief.particles:
            poses = dict(particle.object_poses)
            poses["PEG"] = (
                float(peg_gt.position[0]) + float(rng.normal(0, peg_sigma)),
                float(peg_gt.position[1]) + float(rng.normal(0, peg_sigma)),
                float(peg_gt.position[2]),
                float(peg_gt.orientation[0]),
                float(peg_gt.orientation[1]),
                float(peg_gt.orientation[2]),
                float(peg_gt.orientation[3]),
            )
            poses["NUT"] = (
                float(nut_gt.position[0]),
                float(nut_gt.position[1]),
                float(nut_gt.position[2]),
                float(nut_gt.orientation[0]),
                float(nut_gt.orientation[1]),
                float(nut_gt.orientation[2]),
                float(nut_gt.orientation[3]),
            )
            new_particles.append(
                TabletopState(
                    joint_positions=particle.joint_positions,
                    gripper_open=particle.gripper_open,
                    object_poses=poses,
                )
            )
        n = len(new_particles)
        belief.particles = new_particles
        belief.weights = np.ones(n) / n

    def _run_trial(seed: int, peg_sigma: float) -> bool:
        system = NutAssemblyTAMPSystem(seed=seed, gui=False)
        rng = np.random.default_rng(seed + 999)
        try:
            system.reset(seed=seed)
            symbolic_plan = system.get_symbolic_plan()
            if symbolic_plan is None:
                return False
            ignored = system.get_oig_ignored_objects()
            subsequences = build_object_interaction_graph(symbolic_plan, ignored)
            terminated = False
            for subseq_actions, _ in subsequences:
                goal_add, _ = system.get_subsequence_effects(subseq_actions)
                op = "+".join(a.strip("() ").split()[0] for a in subseq_actions)
                if "assemble" in op:
                    _inject_peg_sigma(system, peg_sigma, rng)
                plan = system.plan_for_goal(goal_add, seed=seed)
                if plan is None:
                    return False
                for action in plan.actions:
                    _, _, terminated, _, _ = system.env.step(action)
                    if terminated:
                        break
                if terminated:
                    break
            return terminated
        finally:
            system.close()

    N_TRIALS = 3

    tight_successes = sum(_run_trial(seed=s, peg_sigma=1e-6) for s in range(N_TRIALS))
    loose_successes = sum(_run_trial(seed=s, peg_sigma=0.05) for s in range(N_TRIALS))

    print(f"\nTight sigma (0.001 mm): {tight_successes}/{N_TRIALS} successes")
    print(f"Loose sigma (50.0 mm): {loose_successes}/{N_TRIALS} successes")


def test_nut_assembly_pointcloud_pca_detection_vs_gt():
    """Point-cloud PCA detection finds NUT and PEG with XY error < 5 cm.

    Yaw is not checked: both objects are rotationally symmetric (round),
    so PCA yaw is arbitrary.
    """
    env = NutAssemblyEnv(gui=False)
    env.reset(seed=7)

    camera_image = env.get_camera_image()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env._get_label_to_id()  # pylint: disable=protected-access
    pcid = env.physics_client_id

    gt_detections = detect_objects_from_segmentation(
        camera_image.segmentation, label_to_id, pcid, detection_pos_std=0.0
    )
    pca_detections = detect_objects_from_pointcloud_pca(
        camera_image.segmentation,
        camera_image.depth,
        label_to_id,
        pcid,
        env.camera_intrinsics,
        camera_pose,
        pixel_noise_std=150.0,
    )

    assert gt_detections, "GT method found nothing"
    assert pca_detections, "PCA method found nothing"

    for label, obj_id in label_to_id.items():
        if label not in pca_detections:
            continue
        assert label in gt_detections, f"{label} detected by PCA but not visible to GT"

        all_points = _unproject_mask_points(
            camera_image.segmentation,
            camera_image.depth,
            obj_id,
            camera_pose,
            env.camera_intrinsics,
        )
        top_face = _extract_top_face_points(all_points)
        effective_std = 150.0 / float(np.sqrt(max(1, len(top_face))))

        gt_pose = gt_detections[label][0]
        pca_pose = pca_detections[label][0]
        gt_xy = np.array(gt_pose[:2])
        pca_xy = np.array(pca_pose[:2])
        xy_error = float(np.linalg.norm(gt_xy - pca_xy))
        assert xy_error < 0.05, f"{label}: XY error {xy_error:.3f} m exceeds 5 cm"
        print(
            f"\n{label}: N_top_face={len(top_face)}, effective_std={effective_std:.2f} px"  # pylint: disable=line-too-long
            f"  GT XY={gt_xy}, PCA XY={pca_xy}, XY err={xy_error:.3f} m"
        )

    env.close()


def test_nut_assembly_top_face_pca_visualization():
    """Visualize top-face 3D points used for PCA pose estimation in nut
    assembly."""
    env = NutAssemblyEnv(gui=False)
    env.reset(seed=42)

    camera_image = env.get_camera_image()
    camera_pose = env.get_camera_pose_se3()
    label_to_id = env._get_label_to_id()  # pylint: disable=protected-access
    pcid = env.physics_client_id

    os.makedirs("videos/perception_test", exist_ok=True)

    print(f"\n{'='*60}")
    print("NUT ASSEMBLY TOP FACE PCA VISUALIZATION")
    print(f"{'='*60}")

    for label, obj_id in label_to_id.items():
        all_points = _unproject_mask_points(
            camera_image.segmentation,
            camera_image.depth,
            obj_id,
            camera_pose,
            env.camera_intrinsics,
        )
        if len(all_points) == 0:
            print(f"\n{label}: not visible")
            continue

        top_face = _extract_top_face_points(all_points)
        gt_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=pcid)

        print(f"\n{label} (id={obj_id}):")
        print(f"  mask points: {len(all_points)}, top-face inliers: {len(top_face)}")
        print(f"  GT pos: ({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f})")

        if len(top_face) >= 3:
            centroid_xy = top_face[:, :2].mean(axis=0)
            print(f"  centroid XY: ({centroid_xy[0]:.3f}, {centroid_xy[1]:.3f})")
            p.addUserDebugPoints(
                top_face.tolist(),
                [[0.0, 0.0, 0.0]] * len(top_face),
                pointSize=5,
                physicsClientId=pcid,
            )

    imageio.imwrite(
        "videos/perception_test/top_face_pca_wrist_nut.png", camera_image.rgb
    )
    frame = capture_image(
        pcid,
        camera_distance=0.9,
        camera_yaw=50,
        camera_pitch=-35,
        camera_target=(0.5, 0.0, 0.0),
        image_width=640,
        image_height=480,
    )
    imageio.imwrite("videos/perception_test/top_face_pca_external_nut.png", frame)
    print("\nSaved: videos/perception_test/top_face_pca_wrist_nut.png")
    print("Saved: videos/perception_test/top_face_pca_external_nut.png")
    print(f"{'='*60}\n")
    # input("Press Enter to close...")
    env.close()
