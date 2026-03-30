"""Shared utilities for loading Objaverse objects into PyBullet."""

from __future__ import annotations

import os
import ssl

import objaverse
import pybullet as p
import trimesh


def setup_ssl_context() -> None:
    """Disable SSL verification for Objaverse downloads."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    ssl._create_default_https_context = (  # pylint: disable=protected-access
        lambda *args, **kwargs: ssl_context
    )


def convert_glb_to_obj(glb_path: str) -> str:
    """Convert a GLB file to OBJ format (cached on disk)."""
    obj_path = glb_path.replace(".glb", ".obj")
    if os.path.exists(obj_path):
        return obj_path
    mesh = trimesh.load(glb_path)
    if hasattr(mesh, "geometry"):
        if len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError(f"No geometry found in {glb_path}")
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.export(obj_path)
    return obj_path


def get_resting_z(obj_id: int, placed_z: float, physics_client_id: int) -> float:
    """Return the world z so the object's bottom rests exactly on the table
    (z=0)."""
    aabb = p.getAABB(obj_id, physicsClientId=physics_client_id)
    local_min_z = float(aabb[0][2]) - placed_z
    return -local_min_z


def load_objaverse_object(
    uid: str,
    scale: float,
    mass: float,
    physics_client_id: int,
) -> int:
    """Download an Objaverse object by UID and load it into PyBullet.

    Returns the PyBullet body ID.
    """
    downloaded = objaverse.load_objects(uids=[uid], download_processes=1)
    glb_path = downloaded[uid]
    obj_path = convert_glb_to_obj(glb_path)
    col_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        physicsClientId=physics_client_id,
    )
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        physicsClientId=physics_client_id,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        physicsClientId=physics_client_id,
    )
