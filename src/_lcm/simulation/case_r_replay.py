"""Shared dynamic operands and mandatory summaries for Case-R forward stages."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.current_iid_replay import CaseRReplayInput, CaseRReplayOutput
from _lcm.egm.current_iid_resources_artifact import (
    OriginalResourcesRow,
    resources_row_payload,
)
from _lcm.engine import Regime
from _lcm.params.edges import regime_kernel_params
from _lcm.typing import FlatParams, PytreeValue, ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import BoolND, ReferenceName


class CaseRReplayProfile(NamedTuple):
    """Checked compiler descriptors, separate from the numeric output contract."""

    actions: Mapping[str, jax.ShapeDtypeStruct]
    value: jax.ShapeDtypeStruct
    physical: Mapping[str, jax.ShapeDtypeStruct]
    law_invalid: jax.ShapeDtypeStruct
    invalid: jax.ShapeDtypeStruct
    feasible: jax.ShapeDtypeStruct


def case_r_profile_output(*, output: ShapeDtypePytree) -> CaseRReplayProfile:
    """Read the actual registered compiler result without numeric type claims."""
    if not isinstance(output, tuple):
        raise ExecutionPlanningError("Case-R profile omitted its output record.")
    children = tuple(output)
    if type(output) is not CaseRReplayOutput or len(children) != len(
        CaseRReplayOutput._fields
    ):
        raise ExecutionPlanningError(
            "Case-R profile has an incompatible output schema."
        )
    actions, value, physical, law, invalid, feasible = children
    return CaseRReplayProfile(
        actions=_profile_mapping(values=actions),
        value=_profile_leaf(value=value),
        physical=_profile_mapping(values=physical),
        law_invalid=_profile_leaf(value=law),
        invalid=_profile_leaf(value=invalid),
        feasible=_profile_leaf(value=feasible),
    )


def _profile_mapping(*, values: ShapeDtypePytree) -> Mapping[str, jax.ShapeDtypeStruct]:
    """Keep each actual named action/frame leaf and refuse non-mapping children."""
    if not isinstance(values, Mapping):
        raise ExecutionPlanningError(
            "Case-R profile requires named action/frame leaves."
        )
    return {name: _profile_leaf(value=value) for name, value in values.items()}


def _profile_leaf(*, value: ShapeDtypePytree) -> jax.ShapeDtypeStruct:
    """Refuse numerical or placeholder leaves in a compiler-only profile."""
    if not isinstance(value, jax.ShapeDtypeStruct):
        raise ExecutionPlanningError("Case-R profile requires compiler descriptors.")
    return value


def case_r_arguments(
    *,
    regime: Regime,
    replay: CaseRReplayInput,
    row: OriginalResourcesRow,
    states: PytreeValue | ShapeDtypePytree,
    keys: PytreeValue | ShapeDtypePytree,
    overrides: PytreeValue | ShapeDtypePytree,
    flat_params: FlatParams,
    period: int,
    period_value: PytreeValue | ShapeDtypePytree,
    age: PytreeValue | ShapeDtypePytree,
) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Keep original row, fixed/runtime parameters and addressed frame explicit."""
    spec = regime.simulation.case_r_replay_spec
    plan = regime.simulation.current_iid_draw_plan
    if spec is None or plan is None:
        raise ExecutionPlanningError("Case-R replay lacks its model-owned plan.")
    replay.require(spec=spec, period=period)
    return MappingProxyType(
        {
            "row": resources_row_payload(row=row),
            "states": states,
            "keys": keys,
            "overrides": overrides,
            "params": {
                **regime.resolved_fixed_params,
                **regime_kernel_params(flat_params, regime_name=regime.name),
            },
            "group_params": {name: flat_params.get(name, {}) for name in plan.bindings},
            "period": period_value,
            "age": age,
        }
    )


def case_r_status(
    *,
    law_invalid: BoolND,
    invalid: BoolND,
    feasible: BoolND,
    in_regime: BoolND,
) -> tuple[BoolND, BoolND, BoolND]:
    """Check mandatory law/row integrity before interpreting action feasibility."""
    return (
        jnp.any(in_regime & law_invalid),
        jnp.any(in_regime & invalid),
        jnp.any(in_regime & ~feasible),
    )
