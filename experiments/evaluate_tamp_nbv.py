"""Evaluate TampNbvApproach with OperatingRegionPredictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
from pybullet_helpers.camera import capture_image

from residual_controllers.approaches import TampNbvApproach, TampNbvConfig
from residual_controllers.benchmarks import TabletopViewOcclusionTAMPSystem
from residual_controllers.operating_region.predictor import OperatingRegionPredictor


def run_episode(
    seed: int,
    num_objects: int,
    config: TampNbvConfig,
    gui: bool,
    max_steps: int,
    video_dir: Path | None,
) -> dict:
    """Run a single evaluation episode."""
    system = TabletopViewOcclusionTAMPSystem(
        seed=seed, gui=gui, num_objects=num_objects
    )
    try:
        obs, info = system.reset(seed=seed)

        writer = None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(video_dir / f"eval_episode_{seed}.mp4"),
                fps=30,
                codec="libx264",
            )

        def capture_frame() -> None:
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

        approach = TampNbvApproach(system, seed=seed, config=config)
        step_result = approach.reset(obs, info)
        capture_frame()
        obs, reward, terminated, truncated, info = system.step(step_result.action)

        for _ in range(max_steps):
            if step_result.terminate:
                break
            if terminated or truncated:
                approach.step(obs, reward, terminated, truncated, info)
                break
            step_result = approach.step(obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = system.step(step_result.action)
            capture_frame()

        if writer is not None:
            writer.close()

        m = approach.get_metrics()
        return {
            "success": m.success,
            "total_steps": m.total_steps,
            "nbv_calls": m.nbv_calls,
            "nbv_steps": m.nbv_steps,
            "nbv_viewpoints_selected": m.nbv_viewpoints_selected,
            "nbv_early_termination": m.nbv_early_termination,
            "tamp_replans": m.tamp_replans,
            "plan_actions": m.plan_actions,
        }
    finally:
        system.close()


def main() -> None:
    """Run evaluation episodes for TampNbvApproach with
    OperatingRegionPredictor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-objects", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--predictor",
        type=str,
        default="data/operating_region_predictor.pkl",
        help="Path to trained OperatingRegionPredictor pickle. "
        "If not provided, falls back to is_object_set_unknown heuristic.",
    )
    parser.add_argument("--nbv-max-viewpoints", type=int, default=5)
    parser.add_argument("--nbv-max-steps-per-viewpoint", type=int, default=40)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos/tamp-nbv-approach",
        help="Directory to save per-episode videos. Skipped if not provided.",
    )
    args = parser.parse_args()

    predictor: OperatingRegionPredictor | None = None
    if args.predictor is not None:
        predictor = OperatingRegionPredictor()
        predictor.load(args.predictor)
        print(f"Loaded predictor from {args.predictor}")
        print(f"  Fitted operators: {predictor.fitted_operators}")
    else:
        print("No predictor provided — using is_object_set_unknown heuristic.")

    config = TampNbvConfig(
        nbv_max_viewpoints=args.nbv_max_viewpoints,
        nbv_max_steps_per_viewpoint=args.nbv_max_steps_per_viewpoint,
        predictor=predictor,
    )

    video_dir = Path(args.video_dir) if args.video_dir else None

    results = []
    for ep in range(args.num_episodes):
        seed = args.seed + ep
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} (seed={seed}) ---")
        r = run_episode(
            seed=seed,
            num_objects=args.num_objects,
            config=config,
            gui=args.gui,
            max_steps=args.max_steps,
            video_dir=video_dir,
        )
        results.append(r)
        print(
            f"  success={r['success']}  steps={r['total_steps']}  "
            f"nbv_calls={r['nbv_calls']}  nbv_steps={r['nbv_steps']}"
        )

    n = len(results)
    label = (
        f"With ORP predictor ({Path(args.predictor).name})"
        if args.predictor
        else "Heuristic (is_object_set_unknown)"
    )
    print(f"\n=== {label} ({n} episodes) ===")
    print(f"  success_rate:           {sum(r['success'] for r in results) / n:.2%}")
    for key in (
        "total_steps",
        "nbv_calls",
        "nbv_steps",
        "nbv_viewpoints_selected",
        "nbv_early_termination",
        "tamp_replans",
        "plan_actions",
    ):
        print(f"  avg {key + ':':30s} {sum(r[key] for r in results) / n:.1f}")


if __name__ == "__main__":
    main()
