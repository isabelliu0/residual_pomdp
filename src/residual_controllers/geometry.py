"""Geometry utilities."""

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose
from scipy.spatial.transform import Rotation


def invert_pose(pose: Pose) -> Pose:
    """Compute the inverse of a pose transform."""
    rot = Rotation.from_quat(pose.orientation)
    inv_rot = rot.inv()
    inv_quat = inv_rot.as_quat()
    inv_position = -inv_rot.apply(pose.position)
    return Pose(tuple(inv_position), tuple(inv_quat))


def get_half_extents_from_aabb(
    body_id: int,
    physics_client_id: int,
    max_tilt_rad: float = 0.5,
) -> tuple[float, float, float]:
    """Get AABB half extents.

    Allows yaw and small roll/pitch (up to max_tilt_rad) since the AABB
    z half-extent remains close enough for contact checks.  Raises if
    the object is severely tilted (e.g. lying on its side).
    """
    pose = get_pose(body_id, physics_client_id)
    qx, qy = pose.orientation[0], pose.orientation[1]
    if not (abs(qx) < max_tilt_rad and abs(qy) < max_tilt_rad):
        raise NotImplementedError(
            f"Object {body_id} has excessive roll/pitch: {pose.orientation}"
        )
    aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=physics_client_id)
    return (
        (aabb_max[0] - aabb_min[0]) / 2,
        (aabb_max[1] - aabb_min[1]) / 2,
        (aabb_max[2] - aabb_min[2]) / 2,
    )
