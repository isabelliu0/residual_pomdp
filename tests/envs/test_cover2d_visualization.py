"""Tests for Cover2DEnv visualization."""

from residual_controllers.envs.cover2d import (
    Action,
    Cover2DConfig,
    Cover2DEnv,
    GripperAction,
    PickController,
    PlaceController,
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
        if action is None:
            action = Action(
                dx=0.0, dy=0.0, dtheta=0.0, gripper_action=GripperAction.NOOP
            )
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
    # import matplotlib.pyplot as plt

    config = Cover2DConfig(
        seed=42,
        num_particles=10,
        initial_robot_x=2.0,
        initial_robot_y=3.0,
        initial_block_x=2.5,
        initial_block_y=3.0,
    )
    env = Cover2DEnv(config)
    belief, _ = env.reset()

    pick_controller = PickController(env.world, target_object_id=0)
    place_controller = PlaceController(
        env.world,
        target_x=env.world.config.goal_region_x + 0.5,
        target_y=env.world.config.goal_region_y + 0.5,
    )

    for _ in range(50):
        if not env.state.gripper_state.is_holding:
            action = pick_controller.get_action(belief)
        else:
            action = place_controller.get_action(belief)

        if action is None:
            break

        belief, _, terminal, _ = env.step(action)
        # ax = env.render(show_belief=True)
        # plt.pause(0.5)
        if terminal:
            break

    # ax = env.render(show_belief=True)
    # assert ax is not None
    # plt.close()
