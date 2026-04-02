"""Data structures for operating region estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

from residual_controllers.operating_region.features import BeliefFeatures


@dataclass
class SubsequenceRecord:
    """Data record for a subsequence of an execution trace."""

    operator_name: str
    relevant_object_labels: list[str]
    features: BeliefFeatures
    success: bool
    source: str = "mode_a"
    metadata: dict = field(default_factory=dict)


@dataclass
class OperatorRecord:
    """Data record for a single operator within a subsequence.

    operator_name is the individual operator (e.g. 'pick'),
    sequence_name is the full joined subsequence it belongs to (e.g.
    'pick+assemble'). features reflect belief state at the moment this
    operator is about to run.
    """

    operator_name: str
    sequence_name: str
    relevant_object_labels: list[str]
    features: BeliefFeatures
    success: bool
    source: str = "mode_a"
    metadata: dict = field(default_factory=dict)
