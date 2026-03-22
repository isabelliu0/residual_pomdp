"""TAMP system adapter for TabletopViewOcclusionEnv."""

from __future__ import annotations

from pybullet_helpers.geometry import get_pose, set_pose

from residual_controllers.benchmarks.tabletop_base_system import TabletopBaseSystem
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_tamp_base import TabletopTypes
from residual_controllers.envs.tabletop_view_occlusion import TabletopViewOcclusionEnv
from residual_controllers.envs.tabletop_view_occlusion_tamp import (
    TabletopAbstractor,
    TabletopPredicates,
    create_tabletop_operators,
    create_tabletop_skills,
)
from residual_controllers.tamp.structs import PlanningComponents


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

    def _get_planning_components(self):  # type: ignore[override]
        assert self._plan_env is not None
        types = TabletopTypes()
        predicates = TabletopPredicates(types)
        operators = create_tabletop_operators(types, predicates)
        abstractor = TabletopAbstractor(self._plan_env, types, predicates)
        components = PlanningComponents(
            types=types.as_set(),
            predicates=predicates,
            operators=operators,
            abstractor=abstractor,
        )
        skills = create_tabletop_skills(types, operators, self._plan_env, abstractor)
        return components, skills

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
