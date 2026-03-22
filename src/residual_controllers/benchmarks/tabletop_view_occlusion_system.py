"""TAMP system adapter for TabletopViewOcclusionEnv."""

from __future__ import annotations

from pybullet_helpers.geometry import get_pose, set_pose

from residual_controllers.benchmarks.tabletop_base_system import TabletopBaseSystem
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_view_occlusion import TabletopViewOcclusionEnv


class TabletopViewOcclusionTAMPSystem(TabletopBaseSystem):
    """TabletopViewOcclusionEnv wrapped as a TAMP system."""

    def __init__(
        self,
        seed: int | None = None,
        gui: bool = False,
        num_objects: int = 1,
        occlusion_prob: float = 0.7,
    ) -> None:
        self.num_objects = num_objects
        self.occlusion_prob = occlusion_prob
        super().__init__(seed=seed, gui=gui)

    def _create_env(self) -> TabletopViewOcclusionEnv:
        return TabletopViewOcclusionEnv(
            gui=self.gui,
            num_objects=self.num_objects,
            occlusion_prob=self.occlusion_prob,
        )

    def _create_plan_env(self) -> TabletopBaseEnv:
        return TabletopViewOcclusionEnv(
            gui=False,
            num_objects=self.num_objects,
            occlusion_prob=self.occlusion_prob,
        )

    def _sync_extra_scene_objects(self) -> None:
        assert self._plan_env is not None
        assert isinstance(self.env, TabletopViewOcclusionEnv)
        assert isinstance(self._plan_env, TabletopViewOcclusionEnv)
        assert self.env.scene is not None
        assert self._plan_env.scene is not None

        target_area_pose = get_pose(
            self.env.scene.target_area_id, self.env.physics_client_id
        )
        set_pose(
            self._plan_env.scene.target_area_id,
            target_area_pose,
            self._plan_env.physics_client_id,
        )

    def get_oig_ignored_objects(self) -> set[str]:
        return {"robot", "table", "target_area"}
