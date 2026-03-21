"""Train operating region predictor from collected SubsequenceRecords."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from residual_controllers.operating_region.features import BeliefFeatures
from residual_controllers.operating_region.predictor import OperatingRegionPredictor
from residual_controllers.operating_region.structs import SubsequenceRecord


def main() -> None:
    """Train operating region predictor from collected SubsequenceRecords."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/subsequence_records.pkl",
        help="Path to collected SubsequenceRecords pickle",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/operating_region_predictor.pkl",
        help="Where to save the trained predictor",
    )
    parser.add_argument(
        "--sigma-threshold",
        type=float,
        default=0.8,
        help="P(success) threshold for find_sigma_threshold()",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(data_path, "rb") as f:
        records: list[SubsequenceRecord] = pickle.load(f)

    print(f"Loaded {len(records)} records from {data_path}")

    predictor = OperatingRegionPredictor()
    predictor.fit(records)

    predictor.save(output_path)
    print(f"Saved predictor to {output_path}")

    print("\n=== Per-operator summary ===")
    for op in predictor.fitted_operators:
        op_records = [r for r in records if r.operator_name == op]
        successes = sum(r.success for r in op_records)
        sigmas = [r.features.relevant_sigma for r in op_records]
        sigma_min, sigma_max = float(np.min(sigmas)), float(np.max(sigmas))

        p_at_zero = predictor.predict(
            op,
            BeliefFeatures(
                sigma_scalar=0.0,
                n_known=1,
                n_unknown=0,
                mean_confidence=1.0,
                relevant_sigma=0.0,
            ),
        )
        p_at_max = predictor.predict(
            op,
            BeliefFeatures(
                sigma_scalar=sigma_max,
                n_known=0,
                n_unknown=1,
                mean_confidence=0.0,
                relevant_sigma=sigma_max,
            ),
        )

        sigma_thresh = predictor.find_sigma_threshold(
            op, threshold=args.sigma_threshold
        )

        print(
            f"  {op}: {len(op_records)} records, "
            f"{successes} successes ({100 * successes // len(op_records)}%), "
            f"sigma=[{sigma_min:.4f}, {sigma_max:.4f}], "
            f"P(success|sigma=0)={p_at_zero:.3f}, "
            f"P(success|sigma=max)={p_at_max:.3f}, "
            f"sigma_threshold(p>={args.sigma_threshold})={sigma_thresh:.4f}"
        )
        if sigma_thresh == 0.0 and p_at_zero < args.sigma_threshold:
            print(
                f"    WARNING: P(success) never reaches {args.sigma_threshold} "
                f"(max={p_at_zero:.3f} at sigma=0). "
                f"Consider lowering --sigma-threshold."
            )


if __name__ == "__main__":
    main()
