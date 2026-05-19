"""Helpers used by the deprecated `validate_transition_probs` public function."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lcm._grids import DiscreteGrid
from lcm.typing import RegimeName, StateName

if TYPE_CHECKING:
    import lcm.api.model
    import lcm.api.regime


def _extract_markov_transition(
    *,
    raw_transition: object,
    state_name: StateName,
    regime_name: RegimeName,
    target_regime_name: RegimeName | None,
) -> lcm.api.regime.MarkovTransition:
    """Extract a MarkovTransition from a raw transition, handling per-target dicts."""
    from lcm.api.regime import MarkovTransition  # noqa: PLC0415

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


def _build_grids(user_regime: lcm.api.regime.Regime) -> dict[str, DiscreteGrid]:
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
    model: lcm.api.model.Model,
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
