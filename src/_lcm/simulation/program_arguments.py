"""Metadata-only call bindings shared by forward dispatch and profiling."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax
import numpy as np

from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_KEYS_ARGUMENT,
    CURRENT_IID_OVERRIDES_ARGUMENT,
)
from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import CoreProgram
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.shock_overrides import CurrentIIDOverride
from _lcm.simulation.operand_placement import SubjectArgumentNames
from _lcm.typing import PRNGKeyND, PytreeValue, QualifiedName, ShapeDtypePytree
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteState,
    ReferenceName,
    ScalarFloat,
    ScalarInt,
    StateName,
)

ACTIVE_SUBJECTS_ARGUMENT = "_lcm_active_subjects"


def declares_active_subjects(*, program: CoreProgram) -> bool:
    """Whether a decision program declares the active-subject mask operand."""
    builder = program.argument_builder
    return (
        isinstance(builder, SubjectArgumentNames)
        and ACTIVE_SUBJECTS_ARGUMENT in builder.subject_arg_names
    )


def decision_arguments(
    *,
    current_iid: Mapping[
        ReferenceName, PytreeValue | ShapeDtypePytree
    ] = MappingProxyType({}),
    states: Mapping[StateName, PytreeValue | ShapeDtypePytree],
    discrete_actions: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    continuous_actions: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    taste_keys: Mapping[ReferenceName, PRNGKeyND | jax.ShapeDtypeStruct],
    next_values: PytreeValue | ShapeDtypePytree,
    references: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    params: Mapping[QualifiedName, PytreeValue | ShapeDtypePytree],
    period: PytreeValue | ShapeDtypePytree,
    age: PytreeValue | ShapeDtypePytree,
    active_subjects: BoolND
    | np.ndarray[tuple[int], np.dtype[np.bool_]]
    | jax.ShapeDtypeStruct
    | None = None,
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Preserve the decision call's existing last-writer precedence exactly."""
    return {
        **states,
        **discrete_actions,
        **continuous_actions,
        **taste_keys,
        **current_iid,
        "next_regime_to_V_arr": next_values,
        **references,
        **params,
        "period": period,
        "age": age,
        **(
            {ACTIVE_SUBJECTS_ARGUMENT: active_subjects}
            if active_subjects is not None
            else {}
        ),
    }


def current_iid_arguments(
    *,
    regime: Regime,
    flat_params: Mapping[str, PytreeValue | ShapeDtypePytree],
    keys: Mapping[str, PRNGKeyND | jax.ShapeDtypeStruct],
    overrides: Mapping[str, CurrentIIDOverride] = MappingProxyType({}),
) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Use the same group/key operands for forward profiling and dispatch."""
    plan = regime.simulation.current_iid_draw_plan
    if plan is None:
        return MappingProxyType({})
    return cast(
        "Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]",
        MappingProxyType(
            {
                CURRENT_IID_KEYS_ARGUMENT: keys,
                CURRENT_IID_OVERRIDES_ARGUMENT: overrides,
                CURRENT_IID_ARGUMENT: {
                    name: flat_params.get(name, {}) for name in plan.bindings
                },
            }
        ),
    )


def process_node_arguments(
    *,
    regime: Regime,
    regimes: Mapping[str, Regime],
    base_spaces: Mapping[str, StateActionSpace],
) -> dict[ReferenceName, ContinuousState | DiscreteState]:
    """Supply resolved solve axes only for this source's node-draw targets."""
    return {
        f"nodes_{target}__next_{state}": nodes
        for target in regime.simulation.transition_plans
        for state, nodes in base_spaces[target].states.items()
        if isinstance(
            process := regimes[target].simulation.grids[state],
            _ContinuousStochasticProcess,
        )
        and process.simulated_draws == "nodes"
    }


def transition_arguments(
    *,
    current_shocks: Mapping[
        ReferenceName, PytreeValue | ShapeDtypePytree
    ] = MappingProxyType({}),
    states: Mapping[StateName, PytreeValue | ShapeDtypePytree],
    carried: Mapping[StateName, PytreeValue | ShapeDtypePytree],
    actions: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    keys: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    period: PytreeValue | ShapeDtypePytree,
    age: PytreeValue | ShapeDtypePytree,
    params: Mapping[QualifiedName, PytreeValue | ShapeDtypePytree],
    process_nodes: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree] | None = None,
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Preserve the state/route call's existing last-writer parameter precedence."""
    return {
        **states,
        **carried,
        **current_shocks,
        **({} if process_nodes is None else process_nodes),
        **actions,
        **keys,
        "period": period,
        "age": age,
        **params,
    }


def policy_prepare_arguments(
    *,
    payload: PytreeValue | ShapeDtypePytree,
    states: Mapping[StateName, PytreeValue] | Mapping[StateName, ShapeDtypePytree],
    params: Mapping[QualifiedName, PytreeValue]
    | Mapping[QualifiedName, ShapeDtypePytree],
    age: ScalarFloat | ScalarInt | jax.ShapeDtypeStruct,
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Bind the dynamic published bank to its declared reconstruction inputs."""
    return {
        "payload": payload,
        # A copy of a concrete or an abstract mapping is that same kind of tree.
        "states": cast(
            "PytreeValue | ShapeDtypePytree", MappingProxyType(dict(states))
        ),
        "params": params,
        "age": age,
    }


def policy_rank_arguments(
    *,
    payload: PytreeValue | ShapeDtypePytree,
    bank: PytreeValue | ShapeDtypePytree,
    canonical_states: Mapping[StateName, PytreeValue]
    | Mapping[StateName, ShapeDtypePytree],
    params: Mapping[QualifiedName, PytreeValue]
    | Mapping[QualifiedName, ShapeDtypePytree],
    age: PytreeValue | ShapeDtypePytree,
    next_values: PytreeValue | ShapeDtypePytree,
    references: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Bind canonical ranking with the dispatch's final reference precedence."""
    return {
        "payload": payload,
        "bank": bank,
        "canonical_states": cast(
            "PytreeValue | ShapeDtypePytree", MappingProxyType(dict(canonical_states))
        ),
        "params": params,
        "age": age,
        "next_regime_to_V_arr": next_values,
        **references,
    }


def gate_fold_arguments(
    *,
    edge_values: PytreeValue | ShapeDtypePytree,
    flat_params: PytreeValue | ShapeDtypePytree,
    fold_age: PytreeValue | ShapeDtypePytree,
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Bind the raw addressed inputs of a gated-continuation fold."""
    return {
        "edge_values": edge_values,
        "flat_params": flat_params,
        "fold_age": fold_age,
    }


def gate_route_arguments(
    *,
    edge_values: PytreeValue | ShapeDtypePytree,
    next_states: PytreeValue | ShapeDtypePytree,
    new_subject_regime_ids: PytreeValue | ShapeDtypePytree,
    subjects_in_regime: PytreeValue | ShapeDtypePytree,
    flat_params: PytreeValue | ShapeDtypePytree,
    own_stakeholder: PytreeValue | ShapeDtypePytree,
    new_own_stakeholder: PytreeValue | ShapeDtypePytree,
    fold_age: PytreeValue | ShapeDtypePytree,
) -> dict[ReferenceName, PytreeValue | ShapeDtypePytree]:
    """Bind realized-state gate operands and the current carrier."""
    return {
        "edge_values": edge_values,
        "next_states": next_states,
        "new_subject_regime_ids": new_subject_regime_ids,
        "subjects_in_regime": subjects_in_regime,
        "flat_params": flat_params,
        "own_stakeholder": own_stakeholder,
        "new_own_stakeholder": new_own_stakeholder,
        "fold_age": fold_age,
    }
