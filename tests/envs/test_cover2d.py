"""Tests for Cover2DEnv visualization."""

import numpy as np

from residual_controllers.envs.cover2d import (
    Action,
    Cover2DConfig,
    Cover2DEnv,
    PickController,
    PlaceController,
    get_mean_state,
)
from residual_controllers.envs.cover2d_tamp import (
    Cover2DAbstractor,
    Cover2DPredicates,
    Cover2DTypes,
    create_cover2d_operators,
    create_cover2d_skills,
)
from residual_controllers.tamp.sesame_runner import run_tamp
from residual_controllers.tamp.structs import PlanningComponents


def test_beacon_visualization():
    """Test visualization of Cover2DEnv with beacons."""
    # # Uncomment to enable visualization
    # import matplotlib.pyplot as plt

    config = Cover2DConfig(
        seed=42,
        num_particles=100,
        num_beacons=3,
    )
    env = Cover2DEnv(config)
    env.reset()

    def compute_belief_std():
        x_vals = [p.robot_pose.x for p in env.belief.particles]
        y_vals = [p.robot_pose.y for p in env.belief.particles]
        return np.std(x_vals), np.std(y_vals)

    nearby = env.world.get_nearby_beacons(env.state)
    print(f"Robot initially near {len(nearby)} beacons")

    # ax = env.render(show_belief=True)

    for step in range(20):
        action = Action(dx=0.5, dy=-0.5, dtheta=0.0)
        _, _, terminal, info = env.step(action)

        nearby = env.world.get_nearby_beacons(env.state)
        std_x, std_y = compute_belief_std()

        beacons_in = info["observation"].beacons_in
        has_pose_obs = info["observation"].robot_pose is not None

        print(
            f"Step {step}: Robot at ({env.state.robot_pose.x:.2f}, {env.state.robot_pose.y:.2f}), "  # pylint: disable=line-too-long
            f"near {len(nearby)} beacons, beacons_in={beacons_in}, "
            f"pose_obs={has_pose_obs}, std=({std_x:.3f}, {std_y:.3f})"
        )

        # ax = env.render(show_belief=True)
        # plt.pause(0.3)
        if terminal:
            break

    # plt.close()

    assert len(env.state.beacon_poses) == 3


def test_visualization_with_controller():
    """Test visualization of Cover2DEnv with PickController."""
    # # Uncomment to enable visualization
    # import matplotlib.pyplot as plt

    config = Cover2DConfig(seed=42, num_particles=100, num_beacons=3)
    env = Cover2DEnv(config)
    belief, _ = env.reset()

    pick_controller = PickController(env.world, target_object_id=0)

    for _ in range(5):
        action = pick_controller.get_action(belief)
        belief, _, terminal, _ = env.step(action)
        # ax = env.render(show_belief=True)
        # plt.pause(0.3)
        if terminal:
            break

    # ax = env.render(show_belief=True)
    # assert ax is not None
    # plt.close()


def test_visualization_full_episode():
    """Test visualization of Cover2DEnv for a full episode."""
    # # Uncomment to enable visualization
    # from pathlib import Path

    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FFMpegWriter

    config = Cover2DConfig(
        seed=0,
        num_particles=10,
        transition_noise_std=0.3,
        num_beacons=3,
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


def test_cover2d_with_tamp():
    """Test Cover2D with bilevel TAMP planner."""
    config = Cover2DConfig(seed=42, num_particles=10, transition_noise_std=0.3)
    env = Cover2DEnv(config)
    belief, _ = env.reset()

    types = Cover2DTypes()
    predicates = Cover2DPredicates(types)
    operators = create_cover2d_operators(types, predicates)
    abstractor = Cover2DAbstractor(env.world, types, predicates)

    objects, init_atoms, goal_atoms = abstractor.reset(belief)
    print(f"Objects: {[o.name for o in objects]}")
    print(f"Initial atoms: {[str(a) for a in init_atoms]}")
    print(f"Goal atoms: {[str(a) for a in goal_atoms]}")

    components = PlanningComponents(
        types=types.as_set(),
        predicates=predicates,
        operators=operators,
        abstractor=abstractor,
    )

    skills = create_cover2d_skills(types, predicates, operators, env.world)

    def transition_fn(_belief_state, action):
        next_belief, _, _, _ = env.step(action)
        return next_belief

    goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

    plan, graph = run_tamp(
        components=components,
        skills=skills,
        initial_state=belief,
        goal=goal,
        state_abstractor=abstractor.step,
        transition_function=transition_fn,
        timeout=30.0,
        seed=42,
    )

    assert plan is not None, "Should find a plan"
    print("\nFound valid TAMP plan!")

    if graph.abstract_action_edges:
        abstract_actions = [action.name for _, action, _ in graph.abstract_action_edges]
        print(f"  - Abstract plan: {' → '.join(abstract_actions)}")

    print(f"  - Refined trajectory: {len(plan.actions)} low-level actions")
