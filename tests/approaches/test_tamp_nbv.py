"""Smoke test for TAMP+NBV approach."""

import os

import imageio
import pytest
from pybullet_helpers.camera import capture_image

from residual_controllers.approaches import TampNbvApproach, TampNbvConfig
from residual_controllers.benchmarks import TabletopPickTAMPSystem


@pytest.mark.skip(reason="Outdated test.")
def test_tamp_nbv():
    """Test the TAMP + NBV approach on the TabletopPickTAMPSystem."""
    system = TabletopPickTAMPSystem(seed=42, gui=False, num_objects=5)
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
