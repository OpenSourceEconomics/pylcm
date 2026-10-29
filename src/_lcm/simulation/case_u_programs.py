"""Model-owned native subject programs for original Case-U reconstruction."""

from dataclasses import replace
from types import MappingProxyType

import jax
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_REPLAY_PROGRAM,
    case_u_future_reads,
)
from _lcm.engine import Regime
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import OutputRoleTree
from _lcm.regime_building.Q_and_F import _BoundFeasibility, _get_feasibility
from _lcm.simulation.case_u_replay import (
    CaseUActionReplay,
    _unflatten_case_u_output,
)
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from lcm.exceptions import ExecutionPlanningError
from lcm.shocks import ShockName
from lcm.typing import ActionName


def _case_u_output_roles(
    *,
    action_name: ActionName,
    physical_names: tuple[ShockName, ...],
    discrete_action_names: tuple[ActionName, ...] = (),
    additional_continuous_action_names: tuple[ActionName, ...] = (),
) -> OutputRoleTree:
    """Map registered compiler descriptors without allocating prototype arrays."""
    floating = jax.ShapeDtypeStruct((), canonical_float_dtype())
    integer = jax.ShapeDtypeStruct((), np.dtype(np.int32))
    boolean = jax.ShapeDtypeStruct((), np.dtype(np.bool_))
    prototype = _unflatten_case_u_output(
        None,
        (
            {
                action_name: floating,
                **dict.fromkeys(discrete_action_names, integer),
                **dict.fromkeys(additional_continuous_action_names, floating),
            },
            floating,
            dict.fromkeys(physical_names, floating),
            boolean,
            boolean,
            boolean,
        ),
    )
    roles = (
        {
            action_name: "action",
            **dict.fromkeys(discrete_action_names, "action"),
            **dict.fromkeys(additional_continuous_action_names, "action"),
        },
        "decision_value",
        dict.fromkeys(physical_names, "current_shock"),
        "current_iid_invalid",
        "current_iid_invalid",
        "action_feasible",
    )
    return jax.tree.unflatten(jax.tree.structure(prototype), jax.tree.leaves(roles))


def declare_case_u_replay_programs(*, regime: Regime) -> Regime:
    spec = regime.simulation.case_u_replay_spec
    if spec is None or not spec.reconstructs_candidates:
        return regime
    plan = regime.simulation.current_iid_draw_plan
    if plan is None or regime.has_taste_shocks:
        raise ExecutionPlanningError("Case-U reconstruction lacks its original frame.")
    decisions = {}
    for period, declaration in spec.periods.items():
        functions, constraints = regime.simulation.feasibility_pools_by_period[period]
        feasibility = _get_feasibility(functions=functions, constraints=constraints)
        if isinstance(feasibility, _BoundFeasibility):
            raise ExecutionPlanningError(
                "Case-U reconstruction cannot erase bounded feasibility status."
            )
        subject_names = ("states", "keys", "overrides")
        decisions[period] = CoreProgram(
            name=CASE_U_REPLAY_PROGRAM,
            function=_SubjectTiled(
                func=CaseUActionReplay(
                    declaration=declaration, draw_plan=plan, feasibility=feasibility
                ),
                subject_arg_names=subject_names,
            ),
            argument_builder=_ArgumentsBoundAtDispatch(
                program_name=CASE_U_REPLAY_PROGRAM, subject_arg_names=subject_names
            ),
            requirements=CoreExecutionRequirements(
                tiled_axes=(subject_axis(state_names=regime.simulation.state_names),),
                value_reads=case_u_future_reads(
                    payload=declaration.future.template(),
                    key=spec.artifact_key,
                    period=period,
                    regime=regime.name,
                    core=CASE_U_REPLAY_PROGRAM,
                    argument="future",
                ),
            ),
            output_roles=_case_u_output_roles(
                action_name=declaration.action_name,
                physical_names=plan.physical_names,
                discrete_action_names=declaration.discrete_action_names,
                additional_continuous_action_names=declaration.additional_continuous_action_names,
            ),
            disposition=CoreExecutionDisposition.PLANNED,
        )
    return replace(
        regime,
        simulation=replace(
            regime.simulation,
            programs=replace(
                regime.simulation.programs,
                decision=MappingProxyType(decisions),
                action_values=MappingProxyType({}),
            ),
        ),
    )
