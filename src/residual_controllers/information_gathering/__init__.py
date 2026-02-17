"""Information gathering module for active perception."""

from residual_controllers.information_gathering.nbv_planner import (
    NBVPlanner,
    ViewpointCandidate,
    compute_object_target_center,
    compute_viewpoint_ig,
)
from residual_controllers.information_gathering.viewpoint_sampler import (
    compute_region_centroid,
    filter_reachable_viewpoints,
    sample_hemisphere_viewpoints,
)

__all__ = [
    "NBVPlanner",
    "ViewpointCandidate",
    "compute_object_target_center",
    "compute_region_centroid",
    "compute_viewpoint_ig",
    "filter_reachable_viewpoints",
    "sample_hemisphere_viewpoints",
]
