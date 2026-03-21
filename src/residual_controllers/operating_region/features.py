"""Belief feature extraction for operating region estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from residual_controllers.beliefs.structs import Belief


@dataclass
class BeliefFeatures:
    """Scalar features extracted from belief for operating region
    prediction."""

    sigma_scalar: float
    n_known: int
    n_unknown: int
    mean_confidence: float
    relevant_sigma: float


def extract_features(
    belief: Belief,
    object_labels: list[str],
    relevant_object_labels: list[str] | None = None,
) -> BeliefFeatures:
    """Extract scalar belief uncertainty features.

    sigma_scalar: mean position std across all tracked objects.
    relevant_sigma: mean position std restricted to relevant_object_labels.
    """
    positions: dict[str, list[list[float]]] = {lbl: [] for lbl in object_labels}
    for particle in belief.particles:
        for lbl in object_labels:
            pose = particle.object_poses.get(lbl)
            if pose is not None:
                positions[lbl].append([pose[0], pose[1], pose[2]])

    def _sigma(lbls: list[str]) -> float:
        stds = []
        for lbl in lbls:
            pts = positions.get(lbl, [])
            if len(pts) >= 2:
                stds.append(float(np.std(pts, axis=0).mean()))
        return float(np.mean(stds)) if stds else 0.0

    sigma_scalar = _sigma(object_labels)
    relevant_sigma = (
        _sigma(relevant_object_labels) if relevant_object_labels else sigma_scalar
    )

    confidences = [belief.object_confidence.get(lbl, 0.0) for lbl in object_labels]
    return BeliefFeatures(
        sigma_scalar=sigma_scalar,
        n_known=len(belief.known_objects),
        n_unknown=len(belief.unknown_objects),
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
        relevant_sigma=relevant_sigma,
    )
