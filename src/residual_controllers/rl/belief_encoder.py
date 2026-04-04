"""General belief-to-observation encoder for RL training."""

from __future__ import annotations

import numpy as np

from residual_controllers.beliefs.structs import Belief


def _pose_std(poses: np.ndarray, ref_pose: np.ndarray, n_fold: int) -> np.ndarray:
    """Compute std of SE3 poses (N, 7) with yaw-symmetry alignment.

    Position std is computed directly. For orientation, each particle's
    yaw is aligned to the closest n-fold equivalent of ref_pose before
    taking std. Returns a (7,) array: [std_x, std_y, std_z, std_qx,
    std_qy, std_qz, std_qw].
    """
    pos_std = np.std(poses[:, :3], axis=0)

    if n_fold <= 1:
        quat_std = np.std(poses[:, 3:], axis=0)
    else:
        period = 2.0 * np.pi / n_fold
        yaws = 2.0 * np.arctan2(poses[:, 5], poses[:, 6])
        ref_yaw = 2.0 * float(np.arctan2(ref_pose[5], ref_pose[6]))

        k_vals = np.arange(n_fold)
        equiv = yaws[:, None] + k_vals[None, :] * period
        diffs = np.abs((equiv - ref_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        aligned_yaws = equiv[np.arange(len(poses)), np.argmin(diffs, axis=1)]

        aligned_quats = np.stack(
            [
                np.zeros(len(poses)),
                np.zeros(len(poses)),
                np.sin(aligned_yaws / 2.0),
                np.cos(aligned_yaws / 2.0),
            ],
            axis=1,
        )
        quat_std = np.std(aligned_quats, axis=0)

    return np.concatenate([pos_std, quat_std])


def encode_belief_for_rl(
    belief: Belief,
    relevant_labels: list[str],
    n_fold: dict[str, int] | int = 1,
) -> np.ndarray:
    """Encode belief into a fixed-size RL observation vector.

    For each relevant object label:
      - MAP particle's full SE3Pose [x, y, z, qx, qy, qz, qw]   (7)
      - SE3 std across particles (yaw-aligned)                  (7)
    Plus MAP particle's gripper_open                            (1)

    Total: len(relevant_labels) * 14 + 1
    """
    map_idx = int(np.argmax(belief.weights))
    map_particle = belief.particles[map_idx]

    parts: list[np.ndarray] = []
    for lbl in relevant_labels:
        lbl_n_fold = n_fold.get(lbl, 1) if isinstance(n_fold, dict) else n_fold

        map_pose = map_particle.object_poses.get(lbl)
        map_se3 = (
            np.array(map_pose, dtype=np.float64)
            if map_pose is not None
            else np.zeros(7)
        )

        poses = np.array(
            [p.object_poses[lbl] for p in belief.particles if lbl in p.object_poses],
            dtype=np.float64,
        )
        std_se3 = (
            _pose_std(poses, map_se3, lbl_n_fold) if len(poses) >= 2 else np.zeros(7)
        )

        parts.append(map_se3)
        parts.append(std_se3)

    parts.append(np.array([map_particle.gripper_open], dtype=np.float64))
    return np.concatenate(parts).astype(np.float32)
