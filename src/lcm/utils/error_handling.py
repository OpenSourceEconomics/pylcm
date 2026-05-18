import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, overload

import jax.numpy as jnp

from lcm.grids import DiscreteGrid
from lcm.typing import FloatND, RegimeName, StateName
from lcm.user_regime import MarkovTransition
from lcm.user_regime import Regime as UserRegime
from lcm.utils.ast_inspection import _get_func_indexing_params

# Genuine circular import: model.py imports from this module at module level.
# The `model` parameter of `validate_transition_probs` is annotated with the
# fully-qualified string `"lcm.model.Model"` so the beartype claw resolves it
# by importing `lcm.model` at first call — long after the import cycle settles
# — rather than at module-init time. Importing `lcm.model` here keeps `lcm` a
# bound name for the type checker.
if TYPE_CHECKING:
    import lcm.model


@overload
def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.model.Model,
    regime_name: RegimeName,
    state_name: StateName,
) -> None: ...


@overload
def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.model.Model,
    regime_name: RegimeName,
    state_name: StateName,
    target_regime_name: RegimeName,
) -> None: ...


@overload
def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.model.Model,
    regime_name: RegimeName,
) -> None: ...


def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.model.Model,
    regime_name: RegimeName,
    state_name: StateName | None = None,
    target_regime_name: RegimeName | None = None,
) -> None:
    """Validate a transition probability array for shape, values, and row sums.

    When `state_name` is provided, validate a state transition probability array.
    When omitted, validate a regime transition probability array.

    For per-target state transitions (where `state_transitions[state_name]` is a
    dict mapping target regime names to `MarkovTransition` instances), pass
    `target_regime_name` to select the specific transition to validate.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`. If `None`,
            validate a regime transition instead.
        target_regime_name: Target regime name for per-target state transitions.
            Required when the state transition is a per-target dict.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    regime = model.user_regimes[regime_name]

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        markov = _extract_markov_transition(
            raw_transition=raw_transition,
            state_name=state_name,
            regime_name=regime_name,
            target_regime_name=target_regime_name,
        )
        func = markov.func
        grids = _build_grids(regime)
        n_outcomes = len(grids[state_name].categories)
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)
        func = regime.transition.func
        grids = _build_grids(regime)
        n_outcomes = len(model.regime_names_to_ids)

    indexing_params = _get_func_indexing_params(
        func=func, array_param_name="probs_array"
    )

    # Cross-check subscript order against signature order
    sig = inspect.signature(func)
    sig_order = [
        p for p in sig.parameters if p != "probs_array" and p in indexing_params
    ]
    if indexing_params != sig_order:
        func_name = getattr(func, "__name__", "<unknown>")
        msg = (
            f"In function '{func_name}', `probs_array` is indexed as "
            f"`probs_array[{', '.join(indexing_params)}]` but the signature "
            f"order is `probs_array[{', '.join(sig_order)}]`."
        )
        raise ValueError(msg)

    expected_shape = _build_expected_shape(
        indexing_params=indexing_params,
        n_outcomes=n_outcomes,
        grids=grids,
        model=model,
    )

    if probs.shape != expected_shape:
        msg = f"Expected shape {expected_shape} but got {probs.shape}."
        raise ValueError(msg)

    if jnp.any(probs < 0) or jnp.any(probs > 1):
        msg = "All values must be in [0, 1]."
        raise ValueError(msg)

    row_sums = jnp.sum(probs, axis=-1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-6):
        msg = "Rows must sum to 1 along the last axis."
        raise ValueError(msg)


def _extract_markov_transition(
    *,
    raw_transition: object,
    state_name: StateName,
    regime_name: RegimeName,
    target_regime_name: RegimeName | None,
) -> MarkovTransition:
    """Extract a MarkovTransition from a raw transition, handling per-target dicts."""
    if isinstance(raw_transition, MarkovTransition):
        return raw_transition

    if isinstance(raw_transition, Mapping):
        if target_regime_name is None:
            targets = sorted(raw_transition.keys())
            msg = (
                f"State '{state_name}' in regime '{regime_name}' uses per-target "
                f"transitions. Pass target_regime_name to select one of: {targets}."
            )
            raise TypeError(msg)
        if target_regime_name not in raw_transition:
            msg = (
                f"Target regime '{target_regime_name}' not found in per-target "
                f"transitions for state '{state_name}' in regime '{regime_name}'. "
                f"Available targets: {sorted(raw_transition.keys())}."
            )
            raise ValueError(msg)
        entry = raw_transition[target_regime_name]  # ty: ignore[invalid-argument-type]
        if not isinstance(entry, MarkovTransition):
            msg = (
                f"Per-target transition for '{target_regime_name}' in state "
                f"'{state_name}' of regime '{regime_name}' is not a "
                f"MarkovTransition. Got {type(entry).__name__}."
            )
            raise TypeError(msg)
        return entry

    msg = (
        f"State '{state_name}' in regime '{regime_name}' is not a "
        f"MarkovTransition. Got {type(raw_transition).__name__}."
    )
    raise TypeError(msg)


def _build_grids(user_regime: UserRegime) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from regime states and actions."""
    return {
        name: grid
        for name, grid in (*user_regime.states.items(), *user_regime.actions.items())
        if isinstance(grid, DiscreteGrid)
    }


def _build_expected_shape(
    *,
    indexing_params: list[str],
    n_outcomes: int,
    grids: dict[str, DiscreteGrid],
    model: lcm.model.Model,
) -> tuple[int, ...]:
    """Compute expected shape for a transition probability array."""
    shape: list[int] = []
    for param_name in indexing_params:
        if param_name == "period":
            shape.append(model.n_periods)
        elif param_name in grids:
            shape.append(len(grids[param_name].categories))
        else:
            msg = (
                f"Cannot determine expected size for parameter '{param_name}'. "
                f"It is not 'period' and not a DiscreteGrid state or action."
            )
            raise ValueError(msg)
    shape.append(n_outcomes)
    return tuple(shape)
