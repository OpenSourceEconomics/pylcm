"""Metadata-only call bindings shared by forward dispatch and profiling."""

from collections.abc import Mapping
from types import MappingProxyType


def decision_arguments(
    *,
    states: Mapping[str, object],
    discrete_actions: Mapping[str, object],
    continuous_actions: Mapping[str, object],
    taste_keys: Mapping[str, object],
    next_values: object,
    references: Mapping[str, object],
    params: Mapping[str, object],
    period: object,
    age: object,
) -> dict[str, object]:
    """Preserve the decision call's existing last-writer precedence exactly."""
    return {
        **states,
        **discrete_actions,
        **continuous_actions,
        **taste_keys,
        "next_regime_to_V_arr": next_values,
        **references,
        **params,
        "period": period,
        "age": age,
    }


def transition_arguments(
    *,
    states: Mapping[str, object],
    carried: Mapping[str, object],
    actions: Mapping[str, object],
    keys: Mapping[str, object],
    period: object,
    age: object,
    params: Mapping[str, object],
) -> dict[str, object]:
    """Preserve the state/route call's existing last-writer parameter precedence."""
    return {
        **states,
        **carried,
        **actions,
        **keys,
        "period": period,
        "age": age,
        **params,
    }


def policy_prepare_arguments(
    *, payload: object, states: Mapping[str, object], params: object, age: object
) -> dict[str, object]:
    """Bind the dynamic published bank to its declared reconstruction inputs."""
    return {
        "payload": payload,
        "states": MappingProxyType(dict(states)),
        "params": params,
        "age": age,
    }


def policy_rank_arguments(
    *,
    payload: object,
    bank: object,
    canonical_states: Mapping[str, object],
    params: object,
    age: object,
    next_values: object,
    references: Mapping[str, object],
) -> dict[str, object]:
    """Bind canonical ranking with the dispatch's final reference precedence."""
    return {
        "payload": payload,
        "bank": bank,
        "canonical_states": MappingProxyType(dict(canonical_states)),
        "params": params,
        "age": age,
        "next_regime_to_V_arr": next_values,
        **references,
    }


def gate_fold_arguments(
    *,
    edge_values: object,
    edge_flags: object,
    flat_params: object,
    fold_age: object,
) -> dict[str, object]:
    """Bind the raw addressed inputs of a gated-continuation fold."""
    return {
        "edge_values": edge_values,
        "edge_flags": edge_flags,
        "flat_params": flat_params,
        "fold_age": fold_age,
    }


def gate_route_arguments(
    *,
    edge_values: object,
    edge_flags: object,
    next_states: object,
    new_subject_regime_ids: object,
    subjects_in_regime: object,
    flat_params: object,
    own_stakeholder: object,
    new_own_stakeholder: object,
    fold_age: object,
) -> dict[str, object]:
    """Bind realized-state gate operands and the current carrier."""
    return {
        "edge_values": edge_values,
        "edge_flags": edge_flags,
        "next_states": next_states,
        "new_subject_regime_ids": new_subject_regime_ids,
        "subjects_in_regime": subjects_in_regime,
        "flat_params": flat_params,
        "own_stakeholder": own_stakeholder,
        "new_own_stakeholder": new_own_stakeholder,
        "fold_age": fold_age,
    }
