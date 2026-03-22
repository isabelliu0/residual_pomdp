"""TAMP system adapter for TabletopObjectOcclusionEnv."""

from __future__ import annotations

from residual_controllers.benchmarks.tabletop_base_system import TabletopBaseSystem
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_object_occlusion import (
    TabletopObjectOcclusionEnv,
)
from residual_controllers.envs.tabletop_object_occlusion_tamp import (
    TabletopPourAbstractor,
    TabletopPourPredicates,
    create_tabletop_pour_operators,
    create_tabletop_pour_skills,
)
from residual_controllers.envs.tabletop_tamp_base import TabletopTypes
from residual_controllers.tamp.structs import PlanningComponents


class TabletopObjectOcclusionTAMPSystem(TabletopBaseSystem):
    """TabletopObjectOcclusionEnv wrapped as a TAMP system."""

    def _create_env(self) -> TabletopObjectOcclusionEnv:
        return TabletopObjectOcclusionEnv(gui=self.gui)

    def _create_plan_env(self) -> TabletopBaseEnv:
        return TabletopObjectOcclusionEnv(gui=False)

    def _get_planning_components(self):  # type: ignore[override]
        assert self._plan_env is not None
        assert isinstance(self._plan_env, TabletopObjectOcclusionEnv)
        types = TabletopTypes()
        predicates = TabletopPourPredicates(types)
        operators = create_tabletop_pour_operators(types, predicates)
        abstractor = TabletopPourAbstractor(self._plan_env, types, predicates)
        components = PlanningComponents(
            types=types.as_set(),
            predicates=predicates,
            operators=operators,
            abstractor=abstractor,
        )
        skills = create_tabletop_pour_skills(
            types, operators, self._plan_env, abstractor
        )
        return components, skills

    def get_oig_ignored_objects(self) -> set[str]:
        return {"robot", "table"}
