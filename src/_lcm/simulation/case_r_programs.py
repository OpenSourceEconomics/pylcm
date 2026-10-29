"""Native declarations for original-row action-valued Case-R reconstruction."""

import dataclasses
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_replay import (
    CASE_R_REPLAY_PROGRAM,
    CaseRActionReplay,
    CaseRReplayOutput,
)
from _lcm.engine import Regime
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import OutputRoleTree
from _lcm.regime_building.Q_and_F import _BoundFeasibility, _get_feasibility
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import ShockName
from lcm.typing import ActionName, BoolND, FloatND, IntND


def _case_r_output_roles(
    *, action_name: ActionName, physical_names: tuple[ShockName, ...]
) -> OutputRoleTree:
    """Map a numeric prototype through the same registered output structure."""
    floating = jnp.asarray(0.0)
    boolean = jnp.zeros((), dtype=jnp.bool_)
    prototype = CaseRReplayOutput(
        actions={action_name: floating},
        value=floating,
        physical=dict.fromkeys(physical_names, floating),
        law_invalid=boolean,
        invalid=boolean,
        feasible=boolean,
    )
    roles = {
        "actions": "action",
        "value": "decision_value",
        "physical": "current_shock",
        "law_invalid": "current_iid_invalid",
        "invalid": "current_iid_invalid",
        "feasible": "action_feasible",
    }

    # keyword-only-exempt: library-callback=jax.tree_util.tree_map_with_path
    def role(
        path: tuple[jax.tree_util.GetAttrKey | jax.tree_util.DictKey, ...],
        _leaf: FloatND | IntND | BoolND,
    ) -> str:
        field = path[0]
        if not isinstance(field, jax.tree_util.GetAttrKey):
            raise TypeError("Case-R output roles require their declared field paths.")
        return roles[field.name]

    return jax.tree_util.tree_map_with_path(role, prototype)


def declare_case_r_replay_programs(*, regime: Regime) -> Regime:
    """Install the real row reader rather than an action-grid decision."""
    spec = regime.simulation.case_r_replay_spec
    if spec is None:
        return regime
    plan = regime.simulation.current_iid_draw_plan
    if plan is None or regime.has_taste_shocks:
        raise ExecutionPlanningError(
            "Case-R replay requires its original current draw plan without tastes."
        )
    if any(bound.kind == "dag" for bound in regime.simulation.post_decision_bounds):
        raise ExecutionPlanningError(
            "Case-R replay requires the producer's literal borrowing floor."
        )
    decisions = {}
    for period, declaration in spec.periods.items():
        functions, constraints = regime.simulation.feasibility_pools_by_period[period]
        feasibility = _get_feasibility(functions=functions, constraints=constraints)
        if isinstance(feasibility, _BoundFeasibility):
            raise ExecutionPlanningError(
                "Case-R replay cannot discard mandatory bounded-feasibility status."
            )
        subject_names = ("states", "keys", "overrides")
        decisions[period] = CoreProgram(
            name=CASE_R_REPLAY_PROGRAM,
            function=_SubjectTiled(
                func=CaseRActionReplay(
                    declaration=declaration, draw_plan=plan, feasibility=feasibility
                ),
                subject_arg_names=subject_names,
            ),
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=CASE_R_REPLAY_PROGRAM,
                subject_arg_names=subject_names,
            ),
            requirements=CoreExecutionRequirements(
                tiled_axes=(subject_axis(state_names=regime.simulation.state_names),),
                value_reads=declaration.row_reads,
            ),
            output_roles=_case_r_output_roles(
                action_name=declaration.action_name, physical_names=plan.physical_names
            ),
            disposition=CoreExecutionDisposition.PLANNED,
        )
    return dataclasses.replace(
        regime,
        simulation=dataclasses.replace(
            regime.simulation,
            programs=dataclasses.replace(
                regime.simulation.programs,
                decision=MappingProxyType(decisions),
                action_values=MappingProxyType({}),
            ),
        ),
    )
