"""Collect SubsequenceRecords."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from residual_controllers.benchmarks import TabletopViewOcclusionTAMPSystem
from residual_controllers.benchmarks.nut_assembly_system import NutAssemblyTAMPSystem
from residual_controllers.operating_region.data_collect import collect_episode

_ENV_DATA_DIRS: dict[str, str] = {
    "tabletop_view_occlusion": "data/tabletop_view_occlusion",
    "tabletop_object_occlusion": "data/tabletop_object_occlusion",
    "nut_assembly": "data/nut_assembly",
}


def main() -> None:
    """Collect SubsequenceRecords data."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="tabletop_view_occlusion",
        choices=list(_ENV_DATA_DIRS),
        help="Environment to collect data from",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nbv-step-counts",
        type=int,
        nargs="+",
        default=[0, 10, 20, 40],
        help="NBV step counts to try per subsequence",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for subsequence records",
    )
    parser.add_argument(
        "--operator-output",
        type=str,
        default=None,
        help="Output path for operator records",
    )
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    env_dir = _ENV_DATA_DIRS[args.env]
    output_path = Path(args.output or f"{env_dir}/subsequence_records.pkl")
    op_output_path = Path(args.operator_output or f"{env_dir}/operator_records.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nbv_step_counts = tuple(args.nbv_step_counts)
    all_records = []
    all_op_records = []

    for ep in range(args.num_episodes):
        seed = args.seed + ep
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} (seed={seed}) ===")
        if args.env == "nut_assembly":
            system = NutAssemblyTAMPSystem(seed=seed, gui=args.gui)
        else:
            system = TabletopViewOcclusionTAMPSystem(seed=seed, gui=args.gui)
        try:
            records, op_records = collect_episode(
                system, seed=seed, nbv_step_counts=nbv_step_counts
            )
            all_records.extend(records)
            all_op_records.extend(op_records)
            print(
                f"  Collected {len(records)} subsequence records, "
                f"{len(op_records)} operator records "
                f"(total so far: {len(all_records)}, {len(all_op_records)})"
            )
        finally:
            system.close()

    with open(output_path, "wb") as f:
        pickle.dump(all_records, f)

    with open(op_output_path, "wb") as f:
        pickle.dump(all_op_records, f)

    print(f"\nSaved {len(all_records)} subsequence records to {output_path}")
    print(f"Saved {len(all_op_records)} operator records to {op_output_path}")

    operators: dict[str, int] = {}
    for r in all_records:
        operators[r.operator_name] = operators.get(r.operator_name, 0) + 1
    for op, count in operators.items():
        successes = sum(r.success for r in all_records if r.operator_name == op)
        print(
            f"  {op}: {count} records, {successes} successes ({100*successes//count}%)"
        )


if __name__ == "__main__":
    main()
