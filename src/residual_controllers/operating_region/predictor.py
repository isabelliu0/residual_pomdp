"""Operating region predictor: per-operator logistic regression on belief features."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from residual_controllers.operating_region.features import BeliefFeatures
from residual_controllers.operating_region.structs import SubsequenceRecord


class OperatingRegionPredictor:
    """Per-operator logistic regression predicting P(success |
    relevant_sigma)."""

    def __init__(self) -> None:
        self._classifiers: dict[str, LogisticRegression] = {}
        self._scalers: dict[str, StandardScaler] = {}

    def fit(self, records: list[SubsequenceRecord]) -> None:
        """Fit logistic regression classifiers for each operator."""
        by_operator: dict[str, tuple[list[list[float]], list[int]]] = {}
        for rec in records:
            X_list, y_list = by_operator.setdefault(rec.operator_name, ([], []))
            X_list.append([rec.features.relevant_sigma])
            y_list.append(int(rec.success))

        for op, (X_list, y_list) in by_operator.items():
            X = np.array(X_list)
            y = np.array(y_list)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_scaled, y)
            self._scalers[op] = scaler
            self._classifiers[op] = clf

    def predict(self, operator_name: str, features: BeliefFeatures) -> float:
        """Predict P(success) for the given operator and belief features."""
        clf = self._classifiers.get(operator_name)
        scaler = self._scalers.get(operator_name)
        if clf is None or scaler is None:
            return 0.5
        x = scaler.transform([[features.relevant_sigma]])
        probs = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        return float(probs[classes.index(1)]) if 1 in classes else 0.0

    def find_sigma_threshold(
        self,
        operator_name: str,
        threshold: float = 0.95,
        sigma_lo: float = 0.0,
        sigma_hi: float = 1.0,
        n_iter: int = 40,
    ) -> float:
        """Return the largest relevant_sigma where P(success) >= threshold."""
        clf = self._classifiers.get(operator_name)
        if clf is None:
            return sigma_lo

        def _prob(sigma: float) -> float:
            f = BeliefFeatures(
                sigma_scalar=sigma,
                n_known=1,
                n_unknown=0,
                mean_confidence=1.0,
                relevant_sigma=sigma,
            )
            return self.predict(operator_name, f)

        lo, hi = sigma_lo, sigma_hi
        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            if _prob(mid) >= threshold:
                lo = mid
            else:
                hi = mid
        return lo

    def save(self, path: str | Path) -> None:
        """Save classifiers and scalers to a file."""
        with open(path, "wb") as f:
            pickle.dump({"classifiers": self._classifiers, "scalers": self._scalers}, f)

    def load(self, path: str | Path) -> None:
        """Load classifiers and scalers from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "classifiers" in data:
            self._classifiers = data["classifiers"]
            self._scalers = data["scalers"]
        else:
            self._classifiers = data
            self._scalers = {}

    @property
    def fitted_operators(self) -> list[str]:
        """List operators for which we have fitted classifiers."""
        return list(self._classifiers.keys())
