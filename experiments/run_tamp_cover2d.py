"""Run TAMP on Cover2D environment."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from residual_controllers.envs.cover2d import Cover2DConfig, Cover2DEnv
from residual_controllers.envs.cover2d_tamp import (
    Cover2DAbstractor,
    Cover2DPredicates,
    Cover2DTypes,
    create_cover2d_operators,
    create_cover2d_skills,
)
from residual_controllers.tamp.sesame_runner import run_tamp
from residual_controllers.tamp.structs import PlanningComponents


def main():
    """Run TAMP on Cover2D."""
    config = Cover2DConfig(
        seed=42,
        num_particles=50,
        transition_noise_std=0.3,
        num_beacons=3,
        info_bonus_weight=0.1,
    )
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

    # Sesame Planner expects Callable[[_X, _U], _X] for transition function
    def transition_fn(_belief_state, action):
        next_belief, _, _, _ = env.step(action)
        return next_belief

    goal = abstractor.create_abstract_goal(goal_atoms, abstractor.step)

    print("\nRunning TAMP planner...")
    plan, _ = run_tamp(
        components=components,
        skills=skills,
        initial_state=belief,
        goal=goal,
        state_abstractor=abstractor.step,
        transition_function=transition_fn,
        timeout=60.0,
        seed=42,
    )

    if plan is None:
        print("No plan found!")
        return

    print(f"\nFound plan with {len(plan.actions)} actions")
    print(f"States: {len(plan.states)}")

    video_path = Path("videos/tamp_cover2d.mp4")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    writer = FFMpegWriter(fps=10)
    writer.setup(fig, str(video_path), dpi=100)

    print("\nGenerating video...")
    for i, belief_state in enumerate(plan.states):
        env.belief = belief_state
        env.state = belief_state.particles[0]
        env.render(ax=ax, show_belief=True)
        writer.grab_frame()
        print(f"Frame {i+1}/{len(plan.states)}")

    writer.finish()
    plt.close(fig)
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
