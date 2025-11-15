"""Tests for Cover2DEnv visualization."""

from residual_controllers.envs.cover2d import (
    Cover2DConfig,
    Cover2DEnv,
    PickController,
    PlaceController,
    get_mean_state,
)


def test_visualization_with_controller():
    """Test visualization of Cover2DEnv with PickController."""
    # Uncomment to enable visualization
    # import matplotlib.pyplot as plt

    config = Cover2DConfig(seed=42, num_particles=100)
    env = Cover2DEnv(config)
    belief, _ = env.reset()

    pick_controller = PickController(env.world, target_object_id=0)

    for _ in range(5):
        action = pick_controller.get_action(belief)
        belief, _, terminal, _ = env.step(action)
        # ax = env.render(show_belief=True)
        # plt.pause(0.5)
        if terminal:
            break

    # ax = env.render(show_belief=True)
    # assert ax is not None
    # plt.close()


def test_visualization_full_episode():
    """Test visualization of Cover2DEnv for a full episode."""
    # Uncomment to enable visualization
    # from pathlib import Path

    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FFMpegWriter

    config = Cover2DConfig(
        seed=0,
        num_particles=10,
        transition_noise_std=0.3,
    )
    env = Cover2DEnv(config)
    belief, _ = env.reset()

    pick_controller = PickController(env.world, target_object_id=0)
    place_controller = PlaceController(
        env.world,
        target_x=env.world.config.goal_region_x + 0.5,
        target_y=env.world.config.goal_region_y + 0.5,
    )

    # video_path = Path(f"videos/cover2d_test_seed{config.seed}.mp4")
    # video_path.parent.mkdir(parents=True, exist_ok=True)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # writer = FFMpegWriter(fps=10)
    # writer.setup(fig, str(video_path), dpi=100)

    # env.render(ax=ax, show_belief=True)
    # writer.grab_frame()

    for _ in range(50):
        mean_state = get_mean_state(belief)
        if not mean_state.gripper_state.is_holding:
            action = pick_controller.get_action(belief)
        else:
            action = place_controller.get_action(belief)

        belief, _, terminal, _ = env.step(action)
        # env.render(ax=ax, show_belief=True)
        # writer.grab_frame()

        if terminal:
            break

    # writer.finish()
    # plt.close(fig)
    # print(f"Video saved to {video_path}")
