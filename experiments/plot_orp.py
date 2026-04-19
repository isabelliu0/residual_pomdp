"""Visualize Operating Region Predictor data and logistic regression fits."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from residual_controllers.operating_region.features import BeliefFeatures
from residual_controllers.operating_region.predictor import OperatingRegionPredictor

_ENV_CONFIGS = {
    "tabletop_view_occlusion": {
        "label": "TabletopViewOcclusion",
        "records_file": "subsequence_records.pkl",
        "predictor_file": "operating_region_predictor.pkl",
    },
    "nut_assembly": {
        "label": "NutAssembly",
        "records_file": "subsequence_records.pkl",
        "predictor_file": "operating_region_predictor.pkl",
    },
}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_operator_data(
    records: list,
) -> dict[str, tuple[list[float], list[int]]]:
    """Group (relevant_sigma, success) by operator_name."""
    by_op: dict[str, tuple[list[float], list[int]]] = {}
    for rec in records:
        sigmas, successes = by_op.setdefault(rec.operator_name, ([], []))
        sigmas.append(rec.features.relevant_sigma)
        successes.append(int(rec.success))
    return by_op


def plot_orp(
    data_dir: str = "data",
    success_threshold: float = 0.8,
    sigma_hi: float = 0.05,
    output: str | None = None,
) -> None:
    """Plot ORP logistic regression fits."""
    data_root = Path(data_dir)

    # Collect (env_label, operator_name, sigmas, successes, predictor) tuples
    panels: list[tuple[str, str, list, list, OperatingRegionPredictor]] = []

    for env_key, cfg in _ENV_CONFIGS.items():
        env_dir = data_root / env_key
        records_path = env_dir / cfg["records_file"]
        predictor_path = env_dir / cfg["predictor_file"]

        if not records_path.exists() or not predictor_path.exists():
            print(f"[skip] Missing files for {env_key}")
            continue

        records = _load_pickle(records_path)
        predictor = OperatingRegionPredictor()
        predictor.load(predictor_path)

        by_op = _get_operator_data(records)
        for op_name, (sigmas, successes) in sorted(by_op.items()):
            if op_name not in predictor.fitted_operators:
                continue
            panels.append((cfg["label"], op_name, sigmas, successes, predictor))

    if not panels:
        print("No data found — check data_dir path.")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)

    legend_handles: list = []

    for ax_idx, (ax, (_, op_name, sigmas, successes, predictor)) in enumerate(
        zip(axes[0], panels)
    ):
        sigmas_arr = np.array(sigmas)
        successes_arr = np.array(successes)
        jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(successes_arr))

        mask_s = successes_arr == 1
        mask_f = successes_arr == 0
        h_succ = ax.scatter(
            sigmas_arr[mask_s],
            successes_arr[mask_s] + jitter[mask_s],
            c="#2ca02c",
            alpha=0.5,
            s=20,
            zorder=2,
            label="Success",
        )
        h_fail = ax.scatter(
            sigmas_arr[mask_f],
            successes_arr[mask_f] + jitter[mask_f],
            c="#d62728",
            alpha=0.5,
            s=20,
            zorder=2,
            label="Failure",
        )

        # logistic regression sigmoid curve
        x_max = (
            max(sigma_hi, float(sigmas_arr.max()) * 1.1)
            if len(sigmas_arr)
            else sigma_hi
        )
        x_plot = np.linspace(0.0, x_max, 300)
        y_plot = np.array(
            [predictor.predict(op_name, _make_features(s)) for s in x_plot]
        )
        (h_sig,) = ax.plot(
            x_plot,
            y_plot,
            color="#1f77b4",
            lw=2,
            zorder=3,
            label=r"$P(\mathrm{success} \mid \sigma)$",
        )

        # threshold lines
        sigma_thresh = predictor.find_sigma_threshold(
            op_name, threshold=success_threshold, sigma_hi=x_max
        )
        h_pthr = ax.axhline(
            success_threshold,
            color="gray",
            linestyle="--",
            lw=1.2,
            label=r"$P = {:.0f}\%$ threshold".format(success_threshold * 100),
        )
        h_sthr = ax.axvline(
            sigma_thresh,
            color="orange",
            linestyle="--",
            lw=1.2,
            label=r"$\sigma^*_\mathrm{base}$ threshold",
        )
        ax.text(
            sigma_thresh,
            1.08,
            r"$\sigma^*_{{\mathrm{{base}}}}\!=\!{:.4f}$".format(sigma_thresh),
            ha="center",
            va="bottom",
            fontsize=14,
            color="orange",
        )

        ax.set_xlim(left=0.0, right=x_max)
        ax.set_ylim(-0.12, 1.18)
        ax.set_xlabel(r"$\sigma$ uncertainty (m)", fontsize=14)
        is_left = ax_idx == 0
        ax.set_ylabel(
            r"$P(\mathrm{success} \mid \sigma)$" if is_left else "", fontsize=14
        )
        if not is_left:
            ax.tick_params(labelleft=False)
        # ax.set_title(
        #     f"{env_label}  —  {op_name}\n"
        #     r"$n=$" + f"{n_total}  ({n_succ} success, {n_total - n_succ} fail)",
        #     fontsize=9,
        # )
        ax.grid(True, alpha=0.3)

        legend_handles = [h_succ, h_fail, h_sig, h_pthr, h_sthr]

    axes[0][-1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.85,
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def _make_features(sigma: float):
    """Minimal BeliefFeatures with only relevant_sigma set."""
    return BeliefFeatures(
        sigma_scalar=sigma,
        n_known=1,
        n_unknown=0,
        mean_confidence=1.0,
        relevant_sigma=sigma,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ORP logistic regression fits.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="P(success) threshold for sigma* (default: 0.8)",
    )
    parser.add_argument(
        "--sigma-hi",
        type=float,
        default=0.015,
        help="Upper x-axis limit for sigma (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save to file instead of showing (e.g. orp.png)",
    )
    args = parser.parse_args()
    plot_orp(
        data_dir=args.data_dir,
        success_threshold=args.threshold,
        sigma_hi=args.sigma_hi,
        output=args.output,
    )
