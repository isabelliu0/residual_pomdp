"""Smoke test for TAMP+NBV approach."""

import os

import imageio
import pytest
from pybullet_helpers.camera import capture_image

from residual_controllers.approaches import TampNbvApproach, TampNbvConfig
from residual_controllers.benchmarks import TabletopViewOcclusionTAMPSystem


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
            camera_distance=1.5,
            camera_yaw=50,
            camera_pitch=-35,
            camera_target=(0.5, 0.0, 0.0),
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


@pytest.mark.skip(reason="Outdated test.")
def test_tamp_nbv():
    """Test the TAMP + NBV approach on the TabletopViewOcclusionTAMPSystem."""
    system = TabletopViewOcclusionTAMPSystem(seed=42, gui=False, num_objects=5)
    obs, info = system.reset(seed=42)

    writer = None
    os.makedirs("videos/tamp-nbv-test", exist_ok=True)
    writer = imageio.get_writer(
        "videos/tamp-nbv-test/tamp_nbv_external.mp4",
        fps=30,
        codec="libx264",
    )

    def capture_frame():
        if writer is None:
            return
        frame = capture_image(
            system.env.physics_client_id,
            camera_distance=1.5,
            camera_yaw=50,
            camera_pitch=-35,
            camera_target=(0.5, 0.0, 0.0),
            image_width=640,
            image_height=480,
        )
        writer.append_data(frame)

    capture_frame()

    config = TampNbvConfig(nbv_max_viewpoints=10, nbv_max_steps_per_viewpoint=30)
    approach = TampNbvApproach(system, seed=42, config=config)

    step_result = approach.reset(obs, info)
    obs, reward, terminated, truncated, info = system.step(step_result.action)
    capture_frame()

    for _ in range(500):
        if step_result.terminate:
            break
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.step(step_result.action)
        capture_frame()
        if terminated or truncated:
            break

    assert approach.metrics.total_steps > 0
    assert approach.metrics.nbv_calls >= 0

    if writer is not None:
        writer.close()

    system.close()
