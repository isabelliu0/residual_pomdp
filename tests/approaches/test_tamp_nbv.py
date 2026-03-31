"""Smoke test for TAMP+NBV approach."""

import os

import imageio
import numpy as np
from pybullet_helpers.camera import capture_image

from residual_controllers.approaches import TampNbvApproach, TampNbvConfig
from residual_controllers.beliefs import get_best_particle_state
from residual_controllers.benchmarks import TabletopViewOcclusionTAMPSystem
from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.operating_region.features import extract_features


class _FixedSigmaThresholdPredictor:
    """Stub predictor that triggers NBV when relevant_sigma exceeds a fixed
    threshold."""

    def __init__(self, sigma_threshold: float) -> None:
        self._sigma_threshold = sigma_threshold

    def find_sigma_threshold(
        self,
        operator_name: str,  # pylint: disable=unused-argument
        threshold: float,  # pylint: disable=unused-argument
    ) -> float:
        """Ignore operator_name and threshold, just return the fixed
        sigma_threshold."""
        return self._sigma_threshold


def test_approach_tabletop():
    """Smoke test for TampNbvApproach on TabletopViewOcclusionTAMPSystem."""
    system = TabletopViewOcclusionTAMPSystem(seed=42, gui=False, num_objects=1)
    obs, info = system.reset(seed=42)

    os.makedirs("videos/tamp-nbv-approach", exist_ok=True)
    writer = imageio.get_writer(
        "videos/tamp-nbv-approach/tamp_nbv_approach.mp4",
        fps=30,
        codec="libx264",
    )

    def capture_frame():
        frame = capture_image(
            system.env.physics_client_id,
            camera_distance=0.9,
            camera_yaw=90,
            camera_pitch=-25,
            camera_target=(0.5, 0.0, 0.15),
            image_width=640,
            image_height=480,
        )
        writer.append_data(frame)

    config = TampNbvConfig(nbv_max_viewpoints=5, nbv_max_steps_per_viewpoint=20)
    approach = TampNbvApproach(system, seed=42, config=config)

    step_result = approach.reset(obs, info)
    capture_frame()
    obs, reward, terminated, truncated, info = system.step(step_result.action)

    for _ in range(200):
        if step_result.terminate:
            break
        if terminated or truncated:
            approach.step(obs, reward, terminated, truncated, info)
            break
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.step(step_result.action)
        capture_frame()

    writer.close()

    metrics = approach.get_metrics()
    print("\n=== TampNbv metrics ===")
    print(f"  total_steps:            {metrics.total_steps}")
    print(f"  nbv_calls:              {metrics.nbv_calls}")
    print(f"  nbv_steps:              {metrics.nbv_steps}")
    print(f"  nbv_viewpoints_selected:{metrics.nbv_viewpoints_selected}")
    print(f"  nbv_viewpoints_reached: {metrics.nbv_viewpoints_reached}")
    print(f"  nbv_early_termination:  {metrics.nbv_early_termination}")
    print(f"  tamp_replans:           {metrics.tamp_replans}")
    print(f"  plan_actions:           {metrics.plan_actions}")
    print(f"  success:                {metrics.success}")

    assert metrics.total_steps > 0
    if metrics.nbv_calls > 0:
        assert metrics.nbv_steps > 0

    system.close()


def test_approach_nut_assembly():
    """Smoke test for TampNbvApproach on NutAssemblyTAMPSystem."""
    system = NutAssemblyTAMPSystem(seed=42, gui=False)
    obs, info = system.reset(seed=42)

    belief = system.env.belief
    assert belief is not None
    assert "NUT" in belief.known_objects
    assert "PEG" in belief.known_objects

    best = get_best_particle_state(belief)
    peg_gt = np.array([0.5, 0.0])
    peg_particle_xy = np.array(best.object_poses["PEG"][:2])
    peg_offset_mm = float(np.linalg.norm(peg_particle_xy - peg_gt)) * 1000
    print(f"\nBest-particle PEG XY offset: {peg_offset_mm:.2f} mm (clearance ~5.6 mm)")

    os.makedirs("videos/tamp-nbv-approach", exist_ok=True)
    writer = imageio.get_writer(
        "videos/tamp-nbv-approach/nut_assembly.mp4",
        fps=30,
        codec="libx264",
    )

    def capture_frame():
        frame = capture_image(
            system.env.physics_client_id,
            camera_distance=0.9,
            camera_yaw=90,
            camera_pitch=-25,
            camera_target=(0.5, 0.0, 0.15),
            image_width=640,
            image_height=480,
        )
        writer.append_data(frame)

    config = TampNbvConfig(nbv_max_viewpoints=5, nbv_max_steps_per_viewpoint=20)
    approach = TampNbvApproach(system, seed=42, config=config)

    step_result = approach.reset(obs, info)
    capture_frame()
    obs, reward, terminated, truncated, info = system.step(step_result.action)

    for _ in range(300):
        if step_result.terminate:
            break
        if terminated or truncated:
            approach.step(obs, reward, terminated, truncated, info)
            break
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.step(step_result.action)
        capture_frame()

    writer.close()

    metrics = approach.get_metrics()
    print("\n=== NutAssembly TampNbv metrics ===")
    print(f"  total_steps:            {metrics.total_steps}")
    print(f"  nbv_calls:              {metrics.nbv_calls}")
    print(f"  nbv_steps:              {metrics.nbv_steps}")
    print(f"  nbv_viewpoints_selected:{metrics.nbv_viewpoints_selected}")
    print(f"  nbv_viewpoints_reached: {metrics.nbv_viewpoints_reached}")
    print(f"  nbv_early_termination:  {metrics.nbv_early_termination}")
    print(f"  tamp_replans:           {metrics.tamp_replans}")
    print(f"  plan_actions:           {metrics.plan_actions}")
    print(f"  success:                {metrics.success}")

    system.close()


def test_approach_nut_assembly_nbv():
    """NBV runs until PEG+NUT particle sigma drops below a fixed threshold."""
    sigma_threshold = 0.00005
    seed = 42

    system = NutAssemblyTAMPSystem(seed=seed, gui=False)
    obs, info = system.reset(seed=seed)

    belief = system.env.belief
    assert belief is not None
    all_labels = list(belief.object_confidence.keys())
    initial_features = extract_features(belief, all_labels, ["NUT", "PEG"])
    print(
        f"\nInitial relevant_sigma: {initial_features.relevant_sigma * 1000:.2f} mm"
        f"  threshold: {sigma_threshold * 1000:.2f} mm"
    )

    os.makedirs("videos/tamp-nbv-approach", exist_ok=True)
    writer = imageio.get_writer(
        "videos/tamp-nbv-approach/nut_assembly_nbv.mp4",
        fps=30,
        codec="libx264",
    )

    def capture_frame():
        frame = capture_image(
            system.env.physics_client_id,
            camera_distance=0.9,
            camera_yaw=90,
            camera_pitch=-25,
            camera_target=(0.5, 0.0, 0.15),
            image_width=640,
            image_height=480,
        )
        writer.append_data(frame)

    config = TampNbvConfig(
        nbv_max_viewpoints=5,
        nbv_max_steps_per_viewpoint=20,
        predictor=_FixedSigmaThresholdPredictor(sigma_threshold=sigma_threshold),  # type: ignore[arg-type] # pylint: disable=line-too-long
    )
    approach = TampNbvApproach(system, seed=seed, config=config)

    step_result = approach.reset(obs, info)
    capture_frame()
    obs, reward, terminated, truncated, info = system.step(step_result.action)

    for _ in range(300):
        if step_result.terminate:
            break
        if terminated or truncated:
            approach.step(obs, reward, terminated, truncated, info)
            break
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.step(step_result.action)
        capture_frame()

    writer.close()

    metrics = approach.get_metrics()
    print("\n=== NutAssembly NBV-sigma metrics ===")
    print(f"  total_steps:            {metrics.total_steps}")
    print(f"  nbv_calls:              {metrics.nbv_calls}")
    print(f"  nbv_steps:              {metrics.nbv_steps}")
    print(f"  nbv_viewpoints_selected:{metrics.nbv_viewpoints_selected}")
    print(f"  nbv_viewpoints_reached: {metrics.nbv_viewpoints_reached}")
    print(f"  nbv_early_termination:  {metrics.nbv_early_termination}")
    print(f"  tamp_replans:           {metrics.tamp_replans}")
    print(f"  plan_actions:           {metrics.plan_actions}")
    print(f"  success:                {metrics.success}")

    system.close()
