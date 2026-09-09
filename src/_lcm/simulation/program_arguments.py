"""Metadata-only call bindings shared by forward dispatch and profiling."""

from collections.abc import Mapping


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
