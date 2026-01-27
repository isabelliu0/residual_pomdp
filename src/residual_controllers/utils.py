"""Utilities."""

from typing import Any, Callable

import numpy as np


def encode_belief_generic(belief: Any, encoder_fn: Callable) -> np.ndarray:
    """Generic belief encoder that uses a custom encoding function."""
    return encoder_fn(belief)


def compute_translational_covariance(
    particles: list, weights: np.ndarray | None = None
) -> np.ndarray:
    """Compute 2x2 translational covariance matrix from particles."""
    if weights is None:
        weights = np.ones(len(particles)) / len(particles)

    positions = np.array([[p.robot_pose.x, p.robot_pose.y] for p in particles])
    mean_pos = np.average(positions, axis=0, weights=weights)
    centered = positions - mean_pos
    cov_xy = (weights[:, None] * centered).T @ centered

    return cov_xy


def compute_angular_variance_circular(
    particles: list, weights: np.ndarray | None = None
) -> float:
    """Compute angular variance using circular statistics: σ²_θ = -2 log R."""
    if weights is None:
        weights = np.ones(len(particles)) / len(particles)

    angles = np.array([p.robot_pose.theta for p in particles])
    c_bar = np.average(np.cos(angles), weights=weights)
    s_bar = np.average(np.sin(angles), weights=weights)
    R = np.sqrt(c_bar**2 + s_bar**2)
    R = np.clip(R, 1e-10, 1.0)
    sigma_theta_sq = -2 * np.log(R)

    return float(sigma_theta_sq)


def compute_se2_information_reward(
    prev_particles: list,
    curr_particles: list,
    prev_weights: np.ndarray | None = None,
    curr_weights: np.ndarray | None = None,
    alpha: float = 1.0,
) -> float:
    """Compute SE(2)-aware information gain reward.

    r_info = log det Σ_xy(t) + α log σ²_θ(t) - [same at t+1]
    """
    cov_xy_prev = compute_translational_covariance(prev_particles, prev_weights)
    cov_xy_curr = compute_translational_covariance(curr_particles, curr_weights)

    det_prev = max(np.linalg.det(cov_xy_prev), 1e-12)
    det_curr = max(np.linalg.det(cov_xy_curr), 1e-12)

    sigma_theta_sq_prev = max(
        compute_angular_variance_circular(prev_particles, prev_weights), 1e-12
    )
    sigma_theta_sq_curr = max(
        compute_angular_variance_circular(curr_particles, curr_weights), 1e-12
    )

    r_info = (np.log(det_prev) + alpha * np.log(sigma_theta_sq_prev)) - (
        np.log(det_curr) + alpha * np.log(sigma_theta_sq_curr)
    )

    return float(r_info)


def compute_translational_uncertainty(
    particles: list, weights: np.ndarray | None = None
) -> float:
    """Compute scalar measure of translational uncertainty: sqrt(det(Σ_xy))."""
    cov_xy = compute_translational_covariance(particles, weights)
    return float(np.sqrt(max(np.linalg.det(cov_xy), 1e-12)))


def compute_angular_uncertainty(
    particles: list, weights: np.ndarray | None = None
) -> float:
    """Compute scalar measure of angular uncertainty: σ_θ."""
    return float(
        np.sqrt(max(compute_angular_variance_circular(particles, weights), 1e-12))
    )
