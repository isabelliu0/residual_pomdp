"""TAMP system adapter for NutAssemblyEnv."""

from __future__ import annotations

from residual_controllers.benchmarks.tabletop_base_system import TabletopBaseSystem
from residual_controllers.envs.nut_assembly_env import NutAssemblyEnv
from residual_controllers.envs.nut_assembly_tamp import (
    NutAssemblyAbstractor,
    NutAssemblyPredicates,
    create_nut_assembly_operators,
    create_nut_assembly_skills,
)
from residual_controllers.envs.tabletop_base import TabletopBaseEnv
from residual_controllers.envs.tabletop_tamp_base import TabletopTypes
from residual_controllers.tamp.structs import PlanningComponents


class NutAssemblyTAMPSystem(TabletopBaseSystem):
    """NutAssemblyEnv wrapped as a TAMP system."""

    def _create_env(self) -> NutAssemblyEnv:
        return NutAssemblyEnv(gui=self.gui)

    def _create_plan_env(self) -> TabletopBaseEnv:
        return NutAssemblyEnv(gui=False)

    def _get_planning_components(self):  # type: ignore[override]
        assert self._plan_env is not None
        assert isinstance(self._plan_env, NutAssemblyEnv)
        types = TabletopTypes()
        predicates = NutAssemblyPredicates(types)
        operators = create_nut_assembly_operators(types, predicates)
        abstractor = NutAssemblyAbstractor(self._plan_env, types, predicates)
        components = PlanningComponents(
            types=types.as_set(),
            predicates=predicates,
            operators=operators,
            abstractor=abstractor,
        )
        skills = create_nut_assembly_skills(
            types, operators, self._plan_env, abstractor
        )
        return components, skills

    def get_oig_ignored_objects(self) -> set[str]:
        return {"robot", "table"}
