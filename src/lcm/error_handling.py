import inspect
from types import MappingProxyType

import jax.numpy as jnp
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidValueFunctionError,
)
from lcm.interfaces import InternalRegime
from lcm.typing import InternalParams, RegimeName, ScalarFloat


def validate_value_function_array(*, V_arr: Array, age: ScalarFloat) -> None:
    """Validate the value function array for NaN values.

    This function checks the value function array for any NaN values. If any such values
    are found, we raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if jnp.any(jnp.isnan(V_arr)):
        raise InvalidValueFunctionError(
            f"The value function array at age {age} contains NaN values. This "
            "may be due to various reasons:\n"
            "- The user-defined functions returned invalid values.\n"
            "- It is impossible to reach an active regime, resulting in NaN regime\n"
            "  transition probabilities in the normalized transition probabilities."
        )


def validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[str, ...],
    regime_name: str,
    period: int,
) -> None:
    """Validate regime transition probabilities.

    Check that probabilities are finite, sum to 1 across all regimes, and that
    inactive regimes have zero probability.

    Args:
        regime_transition_probs: Immutable mapping of regime names to probability
            arrays.
        active_regimes_next_period: Tuple of regime names active in the next period.
        regime_name: Name of the source regime (for error messages).
        period: Current period (for error messages).

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If probabilities are non-finite,
            don't sum to 1, or assign positive probability to inactive regimes.

    """
    all_probs = jnp.stack(list(regime_transition_probs.values()))

    if jnp.any(~jnp.isfinite(all_probs)):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-finite values in regime transition probabilities from "
            f"'{regime_name}' in period {period}. Check the 'next_regime' function "
            f"of the '{regime_name}' regime."
        )

    sum_all = jnp.sum(all_probs, axis=0)
    if not jnp.allclose(sum_all, 1.0):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' in period {period} "
            f"sum to {sum_all} instead of 1.0. Check the 'next_regime' function "
            f"of the '{regime_name}' regime."
        )

    inactive = set(regime_transition_probs) - set(active_regimes_next_period)
    for r in inactive:
        if jnp.any(regime_transition_probs[r] > 0):
            raise InvalidRegimeTransitionProbabilitiesError(
                f"Regime '{r}' is inactive in period {period + 1} but has positive "
                f"transition probability from '{regime_name}' in period {period}. "
                f"Either make '{r}' active or ensure its probability is 0."
            )


def validate_regime_transitions_all_periods(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for all periods before solve.

    For each period (except the last), for each active non-terminal regime, evaluate the
    regime transition function on a representative grid point and check that inactive
    regimes receive zero probability.

    Args:
        internal_regimes: Immutable mapping of regime names to internal regimes.
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If any inactive regime receives
            positive transition probability.

    """
    for period in range(ages.n_periods - 1):
        active_regimes_next_period = tuple(
            name
            for name, regime in internal_regimes.items()
            if period + 1 in regime.active_periods
        )

        for name, internal_regime in internal_regimes.items():
            if period not in internal_regime.active_periods:
                continue
            if internal_regime.terminal:
                continue

            regime_transition_func = (
                internal_regime.regime_transition_probs.solve  # ty: ignore[unresolved-attribute]
            )

            # Build a representative scalar point from the first element of each grid
            state_action_space = internal_regime.state_action_space(
                regime_params=internal_params[name],
            )
            point: dict[str, object] = {}
            for var_name, arr in state_action_space.states.items():
                point[var_name] = arr[0]
            for var_name, arr in state_action_space.actions.items():
                point[var_name] = arr[0]

            # Filter params to only those accepted by the transition function
            accepted_params = set(inspect.signature(regime_transition_func).parameters)
            filtered_params = {
                k: v for k, v in internal_params[name].items() if k in accepted_params
            }

            regime_transition_probs: MappingProxyType[str, Array] = (
                regime_transition_func(
                    **point,
                    **filtered_params,
                    period=period,
                    age=ages.values[period],
                )
            )

            validate_regime_transition_probs(
                regime_transition_probs=regime_transition_probs,
                active_regimes_next_period=active_regimes_next_period,
                regime_name=name,
                period=period,
            )
